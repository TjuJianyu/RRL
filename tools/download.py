import os 

def download_google_drive_cmd(key, target_path):
	prefix = "wget --load-cookies /tmp/cookies.txt "
	mid = f"\"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \'https://docs.google.com/uc?export=download&id={key}\' -O- | sed -rn \'s/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p\')&id={key}\" " 
	suffix = f" -O {target_path} && rm -rf /tmp/cookies.txt"
	cmd = prefix + mid + suffix
	print('######################################')
	print('runing the follow command.....')
	print(cmd)
	print('######################################')
	os.system(cmd)
	print('done')

if not os.path.exists('checkpoints/supervised_pretrain'):
	os.makedirs('checkpoints/supervised_pretrain')

download_google_drive_cmd('1puDJCfUdexV7jc2QDtzT3GIV6bK_a5DS', 'checkpoints/supervised_pretrain/resnet50.tar.gz')
os.system('tar -xzvf checkpoints/supervised_pretrain/resnet50.tar.gz -C checkpoints/supervised_pretrain/')
os.system('rm checkpoints/supervised_pretrain/resnet50.tar.gz')


download_google_drive_cmd('1yxpGox1on8EG-bgh5m96P-HmFdF1FqKV', 'checkpoints/supervised_pretrain/resnet50w2_imagenet1k_supervised.pth.tar')
download_google_drive_cmd('1BMCdWbRp4nUxRQwKux-_BEQS_5TKC2h6', 'checkpoints/supervised_pretrain/resnet50w4_imagenet1k_supervised.pth.tar') 
download_google_drive_cmd('1vC5es1ysSSZOEhkKQWBafjRyLR_oFPgl', 'checkpoints/supervised_pretrain/2resnet50_imagenet1k_supervised.pth.tar')
download_google_drive_cmd('1J3adr3hepZZXyLcncduBi3v6PZLPAEW5', 'checkpoints/supervised_pretrain/4resnet50_imagenet1k_supervised.pth.tar')
download_google_drive_cmd('1iS82WpEWaTqU6I1qbzEB64mttIp1dYDz', 'checkpoints/supervised_pretrain/resnet50_imagenet1k_supervised_distill5.pth.tar')