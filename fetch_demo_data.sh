#!/bin/bash
set -e

urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

MANUAL_MODE=false
if [[ "${1}" == "--manual" ]]; then
	MANUAL_MODE=true
else
	read -p "Use manual download mode? (y/N): " manual_choice
	if [[ "${manual_choice}" =~ ^[Yy]$ ]]; then
		MANUAL_MODE=true
	fi
fi

manual_wait_for_file () {
	local file_path="$1"
	local source_url="$2"
	local note="$3"

	mkdir -p "$(dirname "$file_path")"

	echo -e "\nManual download required"
	echo "File to place: $file_path"
	echo "Download from: $source_url"
	if [[ -n "$note" ]]; then
		echo "$note"
	fi

	while [[ ! -f "$file_path" ]]; do
		read -p "After placing the file, press Enter to continue... " _
		if [[ ! -f "$file_path" ]]; then
			echo "File not found yet at: $file_path"
		fi
	done
}

fetch_file () {
	local file_path="$1"
	local source_url="$2"
	local note="$3"
	shift 3
	local cmd=("$@")

	if [[ -f "$file_path" ]]; then
		echo "Already present: $file_path"
		return 0
	fi

	if [[ "$MANUAL_MODE" == true ]]; then
		manual_wait_for_file "$file_path" "$source_url" "$note"
	else
		"${cmd[@]}"
	fi
}

prompt_redownload () {
	local archive_path="$1"
	local source_url="$2"
	local note="$3"
	local answer=""

	read -p "Extraction/check failed for $archive_path. Redownload this file? (y/N): " answer
	if [[ "$answer" =~ ^[Yy]$ ]]; then
		rm -f "$archive_path"
		fetch_file "$archive_path" "$source_url" "$note" "${@:4}"
		return 0
	fi

	echo "Keeping existing file: $archive_path"
	return 1
}

extract_smplify () {
	local archive='dataset/body_models/smplify.zip'
	local extract_dir='dataset/body_models/smplify'
	local target='dataset/body_models/smpl/SMPL_NEUTRAL.pkl'
	local src='dataset/body_models/smplify/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'

	if [[ -f "$target" ]]; then
		echo "Already present: $target"
		return 0
	fi

	while true; do
		rm -rf "$extract_dir"
		if unzip -o "$archive" -d "$extract_dir"; then
			if [[ -f "$src" ]]; then
				mv "$src" "$target"
				rm -rf "$extract_dir"
				return 0
			fi
			echo "Expected file not found after extraction: $src"
		else
			echo "Failed to extract: $archive"
		fi

		if ! prompt_redownload "$archive" \
			'https://download.is.tue.mpg.de/download.php?domain=smplify&resume=1&sfile=mpips_smplify_public_v2.zip' \
			'Register/login on SMPLify and download mpips_smplify_public_v2.zip manually.' \
			wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplify&resume=1&sfile=mpips_smplify_public_v2.zip' -O './dataset/body_models/smplify.zip' --no-check-certificate --continue; then
			echo "Cannot continue without a valid SMPLify archive."
			exit 1
		fi
	done
}

extract_smpl () {
	local archive='dataset/body_models/smpl.zip'
	local extract_root='dataset/body_models/smpl'
	local src_f='dataset/body_models/smpl/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
	local src_m='dataset/body_models/smpl/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
	local target_f='dataset/body_models/smpl/SMPL_FEMALE.pkl'
	local target_m='dataset/body_models/smpl/SMPL_MALE.pkl'

	if [[ -f "$target_f" && -f "$target_m" ]]; then
		echo "Already present: $target_f"
		echo "Already present: $target_m"
		return 0
	fi

	while true; do
		rm -rf "$extract_root/smpl"
		if unzip -o "$archive" -d "$extract_root"; then
			if [[ -f "$src_f" && -f "$src_m" ]]; then
				mv "$src_f" "$target_f"
				mv "$src_m" "$target_m"
				rm -rf "$extract_root/smpl"
				return 0
			fi
			echo "Expected files not found after extraction: $src_f or $src_m"
		else
			echo "Failed to extract: $archive"
		fi

		if ! prompt_redownload "$archive" \
			'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip' \
			'Register/login on SMPL and download SMPL_python_v.1.0.0.zip manually.' \
			wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip' -O './dataset/body_models/smpl.zip' --no-check-certificate --continue; then
			echo "Cannot continue without a valid SMPL archive."
			exit 1
		fi
	done
}

extract_tar_if_needed () {
	local archive_path="$1"
	local check_path="$2"
	local source_url="$3"
	local note="$4"
	shift 4
	local cmd=("$@")

	if [[ -e "$check_path" ]]; then
		echo "Already present: $check_path"
		return 0
	fi

	while true; do
		if tar -xf "$archive_path"; then
			if [[ -e "$check_path" ]]; then
				return 0
			fi
			echo "Extraction succeeded but expected path missing: $check_path"
		else
			echo "Failed to extract: $archive_path"
		fi

		if ! prompt_redownload "$archive_path" "$source_url" "$note" "${cmd[@]}"; then
			echo "Cannot continue without a valid archive: $archive_path"
			exit 1
		fi
	done
}

# SMPL Neutral model
echo -e "\nYou need to register at https://smplify.is.tue.mpg.de"
mkdir -p dataset/body_models/smpl
if [[ "$MANUAL_MODE" == true ]]; then
	fetch_file './dataset/body_models/smplify.zip' \
		'https://download.is.tue.mpg.de/download.php?domain=smplify&resume=1&sfile=mpips_smplify_public_v2.zip' \
		'Register/login on SMPLify and download mpips_smplify_public_v2.zip manually.'
else
	read -p "Username (SMPLify):" username
	read -p "Password (SMPLify):" password
	username=$(urle "$username")
	password=$(urle "$password")
	fetch_file './dataset/body_models/smplify.zip' \
		'https://download.is.tue.mpg.de/download.php?domain=smplify&resume=1&sfile=mpips_smplify_public_v2.zip' \
		'' \
		wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplify&resume=1&sfile=mpips_smplify_public_v2.zip' -O './dataset/body_models/smplify.zip' --no-check-certificate --continue
fi
extract_smplify

# SMPL Male and Female model
echo -e "\nYou need to register at https://smpl.is.tue.mpg.de"
if [[ "$MANUAL_MODE" == true ]]; then
	fetch_file './dataset/body_models/smpl.zip' \
		'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip' \
		'Register/login on SMPL and download SMPL_python_v.1.0.0.zip manually.'
else
	read -p "Username (SMPL):" username
	read -p "Password (SMPL):" password
	username=$(urle "$username")
	password=$(urle "$password")
	fetch_file './dataset/body_models/smpl.zip' \
		'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip' \
		'' \
		wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip' -O './dataset/body_models/smpl.zip' --no-check-certificate --continue
fi
extract_smpl

# Auxiliary SMPL-related data
fetch_file 'dataset/body_models.tar.gz' \
	'https://drive.google.com/uc?id=1pbmzRbWGgae6noDIyQOnohzaVnX_csUZ&export=download&confirm=t' \
	'' \
	wget "https://drive.google.com/uc?id=1pbmzRbWGgae6noDIyQOnohzaVnX_csUZ&export=download&confirm=t" -O 'dataset/body_models.tar.gz'
extract_tar_if_needed 'dataset/body_models.tar.gz' 'dataset/body_models/J_regressor_h36m.npy' \
	'https://drive.google.com/uc?id=1pbmzRbWGgae6noDIyQOnohzaVnX_csUZ&export=download&confirm=t' \
	'' \
	tar -xvf dataset/body_models.tar.gz -C dataset/

# Checkpoints
mkdir -p checkpoints
fetch_file 'checkpoints/wham_vit_w_3dpw.pth.tar' \
	'https://drive.google.com/uc?id=1i7kt9RlCCCNEW2aYaDWVr-G778JkLNcB&export=download&confirm=t' \
	'' \
	gdown "https://drive.google.com/uc?id=1i7kt9RlCCCNEW2aYaDWVr-G778JkLNcB&export=download&confirm=t" -O 'checkpoints/wham_vit_w_3dpw.pth.tar'
fetch_file 'checkpoints/wham_vit_bedlam_w_3dpw.pth.tar' \
	'https://drive.google.com/uc?id=19qkI-a6xuwob9_RFNSPWf1yWErwVVlks&export=download&confirm=t' \
	'' \
	gdown "https://drive.google.com/uc?id=19qkI-a6xuwob9_RFNSPWf1yWErwVVlks&export=download&confirm=t" -O 'checkpoints/wham_vit_bedlam_w_3dpw.pth.tar'
fetch_file 'checkpoints/hmr2a.ckpt' \
	'https://drive.google.com/uc?id=1J6l8teyZrL0zFzHhzkC7efRhU0ZJ5G9Y&export=download&confirm=t' \
	'' \
	gdown "https://drive.google.com/uc?id=1J6l8teyZrL0zFzHhzkC7efRhU0ZJ5G9Y&export=download&confirm=t" -O 'checkpoints/hmr2a.ckpt'
fetch_file 'checkpoints/dpvo.pth' \
	'https://drive.google.com/uc?id=1kXTV4EYb-BI3H7J-bkR3Bc4gT9zfnHGT&export=download&confirm=t' \
	'' \
	gdown "https://drive.google.com/uc?id=1kXTV4EYb-BI3H7J-bkR3Bc4gT9zfnHGT&export=download&confirm=t" -O 'checkpoints/dpvo.pth'
fetch_file 'checkpoints/yolov8x.pt' \
	'https://drive.google.com/uc?id=1zJ0KP23tXD42D47cw1Gs7zE2BA_V_ERo&export=download&confirm=t' \
	'' \
	gdown "https://drive.google.com/uc?id=1zJ0KP23tXD42D47cw1Gs7zE2BA_V_ERo&export=download&confirm=t" -O 'checkpoints/yolov8x.pt'
fetch_file 'checkpoints/vitpose-h-multi-coco.pth' \
	'https://drive.google.com/uc?id=1xyF7F3I7lWtdq82xmEPVQ5zl4HaasBso&export=download&confirm=t' \
	'' \
	gdown "https://drive.google.com/uc?id=1xyF7F3I7lWtdq82xmEPVQ5zl4HaasBso&export=download&confirm=t" -O 'checkpoints/vitpose-h-multi-coco.pth'

# Demo videos
fetch_file 'examples.tar.gz' \
	'https://drive.google.com/uc?id=1KjfODCcOUm_xIMLLR54IcjJtf816Dkc7&export=download&confirm=t' \
	'' \
	gdown "https://drive.google.com/uc?id=1KjfODCcOUm_xIMLLR54IcjJtf816Dkc7&export=download&confirm=t" -O 'examples.tar.gz'
extract_tar_if_needed 'examples.tar.gz' 'examples' \
	'https://drive.google.com/uc?id=1KjfODCcOUm_xIMLLR54IcjJtf816Dkc7&export=download&confirm=t' \
	'' \
	tar -xvf examples.tar.gz

