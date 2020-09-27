
function evaluation {
	# file path
	mode=$1
	ep=$2
	save_dir='gan'
	IS_dir='/hdd2/tera/stat_real'
	FID_dir='Get_ImageNet_FID'
	img_dir='/hdd2/tera/ImageNet_1000/out_img'
	z_path='/hdd2/tera/ImageNet_1000/all_noises.npy'
	lb_path='/hdd2/tera/ImageNet_1000/all_labels.npy'
	
	cmd=(python3 main.py 
		--mode ${mode}
		--test_epoch ${ep}
		--save_dir ${save_dir} 
		--real_incep_stat_dir ${IS_dir} 
		--real_fid_stat_dir ${FID_dir} 
		--image_dir ${img_dir} 
		--z_path ${z_path} 
		--label_path ${lb_path}
		--use_numpy_fid False)
		
	echo "Mode: $1"
	CUDA_VISIBLE_DEVICES=0 ${cmd[@]} | tee "${save_dir}/${mode}-log-${ep}.txt"
}


# eval TinyGAN
evaluation test_inter 30
evaluation test_is 30
evaluation test_intra_all 30
