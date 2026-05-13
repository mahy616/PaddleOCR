//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//

#include "src/api/pipelines/ocr.h"
#include "src/pipelines/ocr/result.h"
#include "src/utils/utility.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct OCRConfigCompat {
  const char *det_model_dir;
  const char *rec_model_dir;
  const char *cls_model_dir;
  const char *dict_path;
  bool use_angle_cls;
  bool use_gpu;
};

struct OCRResultCompat {
  std::vector<std::vector<int>> box;
  std::string text;
  double score;
};

namespace {

void LogCompatError(const std::string &message) {
  std::string line = "[ppocr_compat] " + message + "\n";
  OutputDebugStringA(line.c_str());
  std::ofstream out("ppocr_compat_error.log", std::ios::app);
  if (out.is_open()) {
    out << line;
  }
}

std::string Trim(std::string value) {
  const char *space = " \t\r\n";
  size_t first = value.find_first_not_of(space);
  if (first == std::string::npos) {
    return "";
  }
  size_t last = value.find_last_not_of(space);
  value = value.substr(first, last - first + 1);
  if (value.size() >= 2 &&
      ((value.front() == '"' && value.back() == '"') ||
       (value.front() == '\'' && value.back() == '\''))) {
    value = value.substr(1, value.size() - 2);
  }
  return value;
}

std::string JoinPath(const std::string &dir, const std::string &file) {
  if (dir.empty()) {
    return file;
  }
  char last = dir.back();
  if (last == '\\' || last == '/') {
    return dir + file;
  }
  return dir + "\\" + file;
}

bool Exists(const std::string &path) { return Utility::FileExists(path).ok(); }

std::string DirName(const std::string &path) {
  size_t pos = path.find_last_of("\\/");
  if (pos == std::string::npos) {
    return "";
  }
  return path.substr(0, pos);
}

std::string GetModuleDir() {
  HMODULE module = nullptr;
  BOOL ok = GetModuleHandleExA(
      GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
          GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
      reinterpret_cast<LPCSTR>(&GetModuleDir), &module);
  if (!ok || module == nullptr) {
    return "";
  }

  char path[MAX_PATH] = {0};
  DWORD len = GetModuleFileNameA(module, path, MAX_PATH);
  if (len == 0 || len >= MAX_PATH) {
    return "";
  }
  return DirName(path);
}

std::string FindPipelineConfig() {
  std::string module_dir = GetModuleDir();
  std::vector<std::string> candidates;
  if (!module_dir.empty()) {
    candidates.push_back(JoinPath(JoinPath(module_dir, "configs"), "OCR.yaml"));
    candidates.push_back(JoinPath(JoinPath(module_dir, "configs"), "OCR.yml"));
    candidates.push_back(JoinPath(module_dir, "OCR.yaml"));
    candidates.push_back(JoinPath(module_dir, "OCR.yml"));
  }
  candidates.push_back(JoinPath(JoinPath(".", "configs"), "OCR.yaml"));
  candidates.push_back(JoinPath(JoinPath(".", "configs"), "OCR.yml"));

  for (const auto &candidate : candidates) {
    if (Exists(candidate)) {
      return candidate;
    }
  }
  return "";
}

bool HasModelFiles(const std::string &model_dir) {
  bool has_model = Exists(JoinPath(model_dir, "inference.json")) ||
                   Exists(JoinPath(model_dir, "inference.pdmodel"));
  bool has_params = Exists(JoinPath(model_dir, "inference.pdiparams"));
  bool has_config = Exists(JoinPath(model_dir, "inference.yml")) ||
                    Exists(JoinPath(model_dir, "inference.yaml"));
  return has_model && has_params && has_config;
}

std::string ReadModelName(const std::string &model_dir) {
  std::string yaml_path = JoinPath(model_dir, "inference.yml");
  if (!Exists(yaml_path)) {
    yaml_path = JoinPath(model_dir, "inference.yaml");
  }

  std::ifstream in(yaml_path.c_str());
  std::string line;
  const std::string key = "model_name:";
  while (std::getline(in, line)) {
    size_t pos = line.find(key);
    if (pos == std::string::npos) {
      continue;
    }
    return Trim(line.substr(pos + key.size()));
  }
  return "";
}

struct CompatibleOCREngine {
  explicit CompatibleOCREngine(const PaddleOCRParams &params)
      : ocr(new PaddleOCR(params)) {}

  std::unique_ptr<PaddleOCR> ocr;
  std::mutex mutex;
};

bool BuildParams(const OCRConfigCompat *config, PaddleOCRParams *params) {
  if (config == nullptr || params == nullptr || config->det_model_dir == nullptr ||
      config->rec_model_dir == nullptr) {
    return false;
  }

  std::string det_dir = config->det_model_dir;
  std::string rec_dir = config->rec_model_dir;
  if (!HasModelFiles(det_dir) || !HasModelFiles(rec_dir)) {
    return false;
  }

  std::string det_name = ReadModelName(det_dir);
  std::string rec_name = ReadModelName(rec_dir);
  if (det_name.empty() || rec_name.empty()) {
    return false;
  }

  params->text_detection_model_dir = det_dir;
  params->text_detection_model_name = det_name;
  params->text_recognition_model_dir = rec_dir;
  params->text_recognition_model_name = rec_name;
  params->use_doc_orientation_classify = false;
  params->use_doc_unwarping = false;
  params->use_textline_orientation = false;
  params->device = config->use_gpu ? "gpu:0" : "cpu";
  params->thread_num = 1;
  params->cpu_threads = 8;
  params->text_det_limit_side_len = 960;
  params->text_det_limit_type = "max";
  params->text_det_thresh = 0.3f;
  params->text_det_box_thresh = 0.6f;
  params->text_det_unclip_ratio = 1.5f;
  params->text_rec_score_thresh = 0.0f;

  std::string pipeline_config = FindPipelineConfig();
  if (pipeline_config.empty()) {
    LogCompatError(
        "InitializeOCR: missing PaddleOCR pipeline config configs\\OCR.yaml");
    return false;
  }
  params->paddlex_config = Utility::PaddleXConfigVariant(pipeline_config);

  if (config->use_angle_cls && config->cls_model_dir != nullptr &&
      config->cls_model_dir[0] != '\0') {
    std::string cls_dir = config->cls_model_dir;
    if (!HasModelFiles(cls_dir)) {
      return false;
    }
    std::string cls_name = ReadModelName(cls_dir);
    if (cls_name.empty()) {
      return false;
    }
    params->use_textline_orientation = true;
    params->textline_orientation_model_dir = cls_dir;
    params->textline_orientation_model_name = cls_name;
  }

  return true;
}

void FillResult(const OCRPipelineResult &pipeline_result,
                std::vector<OCRResultCompat *> *out) {
  size_t count = pipeline_result.rec_texts.size();
  count = std::min(count, pipeline_result.rec_scores.size());
  count = std::min(count, pipeline_result.rec_polys.size());

  for (size_t i = 0; i < count; ++i) {
    OCRResultCompat *item = new OCRResultCompat();
    item->text = pipeline_result.rec_texts[i];
    item->score = pipeline_result.rec_scores[i];

    const auto &poly = pipeline_result.rec_polys[i];
    size_t point_count = std::min<size_t>(poly.size(), 4);
    item->box.reserve(point_count);
    for (size_t j = 0; j < point_count; ++j) {
      item->box.push_back({static_cast<int>(std::lround(poly[j].x)),
                           static_cast<int>(std::lround(poly[j].y))});
    }
    out->push_back(item);
  }
}

} // namespace

extern "C" __declspec(dllexport) void *
InitializeOCR(const OCRConfigCompat *config) {
  try {
    PaddleOCRParams params;
    if (!BuildParams(config, &params)) {
      LogCompatError("InitializeOCR: invalid model config or missing model files");
      return nullptr;
    }
    return new CompatibleOCREngine(params);
  } catch (const std::exception &e) {
    LogCompatError(std::string("InitializeOCR exception: ") + e.what());
    return nullptr;
  } catch (...) {
    LogCompatError("InitializeOCR unknown exception");
    return nullptr;
  }
}

extern "C" __declspec(dllexport) void
ProcessImage(void *engine_ptr, cv::Mat *image, OCRResultCompat ***results,
             int *result_count) {
  if (results != nullptr) {
    *results = nullptr;
  }
  if (result_count != nullptr) {
    *result_count = 0;
  }
  if (engine_ptr == nullptr || image == nullptr || image->empty() ||
      results == nullptr || result_count == nullptr) {
    return;
  }

  try {
    CompatibleOCREngine *engine =
        static_cast<CompatibleOCREngine *>(engine_ptr);
    std::vector<OCRResultCompat *> items;
    {
      std::lock_guard<std::mutex> lock(engine->mutex);
      auto predict_results = engine->ocr->Predict(*image);
      for (const auto &base_result : predict_results) {
        const OCRResult *ocr_result =
            dynamic_cast<const OCRResult *>(base_result.get());
        if (ocr_result != nullptr) {
          FillResult(ocr_result->GetPipelineResult(), &items);
        }
      }
    }

    if (!items.empty()) {
      OCRResultCompat **result_array = new OCRResultCompat *[items.size()];
      for (size_t i = 0; i < items.size(); ++i) {
        result_array[i] = items[i];
      }
      *results = result_array;
      *result_count = static_cast<int>(items.size());
    }
  } catch (const std::exception &e) {
    LogCompatError(std::string("ProcessImage exception: ") + e.what());
  } catch (...) {
    LogCompatError("ProcessImage unknown exception");
  }
}

extern "C" __declspec(dllexport) void
FreeMemory(OCRResultCompat **results, int result_count) {
  if (results == nullptr) {
    return;
  }
  for (int i = 0; i < result_count; ++i) {
    delete results[i];
  }
  delete[] results;
}

extern "C" __declspec(dllexport) void ReleaseOCR(void *engine_ptr) {
  delete static_cast<CompatibleOCREngine *>(engine_ptr);
}
