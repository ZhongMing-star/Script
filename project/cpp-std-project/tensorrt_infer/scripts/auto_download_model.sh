#!/usr/bin/env bash

set -euo pipefail

# 0. 如果 pt 模型不存在，则下载 pt 模型
MODEL_URL="https://drive.google.com/file/d/13LGmHwemH_jRrTvJTaHk2tla4gObnnRo/view?usp=drive_link"
FILE_ID="13LGmHwemH_jRrTvJTaHk2tla4gObnnRo"

# 默认文件名可通过环境变量覆盖，例如：MODEL_NAME=xxx.pt ./auto_download_model.sh
MODEL_NAME="${MODEL_NAME:-fall_detection.pt}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${ROOT_DIR}/resource/models"
MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"

mkdir -p "${MODEL_DIR}"

if [[ -f "${MODEL_PATH}" ]]; then
	echo "[skip] PT model already exists: ${MODEL_PATH}"
	exit 0
fi

echo "[info] PT model not found, downloading to: ${MODEL_PATH}"

download_with_gdown() {
	if ! command -v gdown >/dev/null 2>&1; then
		return 1
	fi
	gdown --fuzzy "${MODEL_URL}" -O "${MODEL_PATH}"
}

download_with_wget() {
	if ! command -v wget >/dev/null 2>&1; then
		return 1
	fi

	local cookie_file
	local confirm_token
	local tmp_path
	cookie_file="$(mktemp)"
	tmp_path="${MODEL_PATH}.tmp"

	confirm_token="$(wget --quiet --save-cookies "${cookie_file}" --keep-session-cookies \
		"https://drive.google.com/uc?export=download&id=${FILE_ID}" -O- \
		| sed -n 's/.*confirm=\([0-9A-Za-z_]*\).*/\1/p' | head -n1)"

	if [[ -n "${confirm_token}" ]]; then
		wget --load-cookies "${cookie_file}" \
			"https://drive.google.com/uc?export=download&confirm=${confirm_token}&id=${FILE_ID}" \
			-O "${tmp_path}"
	else
		wget "https://drive.google.com/uc?export=download&id=${FILE_ID}" -O "${tmp_path}"
	fi

	rm -f "${cookie_file}"

	if [[ ! -s "${tmp_path}" ]]; then
		rm -f "${tmp_path}"
		return 1
	fi

	mv "${tmp_path}" "${MODEL_PATH}"
}

if download_with_gdown || download_with_wget; then
	echo "[ok] Download finished: ${MODEL_PATH}"
else
	echo "[error] Download failed. Please install gdown or wget, then retry."
	echo "        URL: ${MODEL_URL}"
	exit 1
fi