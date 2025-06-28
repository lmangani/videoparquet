import subprocess

def test_ffmpeg_features():
    print('\n===== ffmpeg -version =====')
    print(subprocess.check_output(['ffmpeg', '-version']).decode())
    print('\n===== ffmpeg -h encoder=ffv1 =====')
    print(subprocess.check_output(['ffmpeg', '-h', 'encoder=ffv1']).decode()) 