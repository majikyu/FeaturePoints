# �T�v
openCV��p���Ă��摜�̓����_�𒊏o����Python�v���O�����ł��D�e�A���S���Y����p�����ۂɌ���������_�̐����o�͂ł��܂��D
# �g����
## opencv�̃C���X�g�[��
opencv��matplotlib��python3�ɃC���X�g�[�����Ă��������D

```
pip3 install opencv-python
```

## �R�}���h�̎��s
### �摜�̏���
�����_�𒊏o����摜���������܂��D

�f�t�H���g�ł́A���݂̃f�B���N�g��������`images`�Ƃ����f�B���N�g���Ɋ܂܂��摜���璊�o���s���܂��D

���p�ł���`���́A`bmp`, `jpg`, `png`�ł��D

### �v���O�����̎��s
�v���O���������s���܂��D
```
python3 fp.py
```
��L�R�}���h�ɂ��A`./images`���̑S�摜��ǂݍ��݁CORB�A���S���Y���ɂ�蔭�����ꂽ�����_��`�悵���摜��`./output`�ɏo�͂���܂��D

�܂��A
```
python3 fp.py -c
```
��L�̂悤��`-c`�I�v�V������t�^���邱�Ƃ�`count���[�h`�ƂȂ�C�W��ނ̃A���S���Y���œ����_�𒊏o�����ۂɔ������������_�̐���`result.csv`�ɏo�͂��܂��D

�S�I�v�V�����ɂ��Ă͎��͂ŏ����܂��B

## �I�v�V����

|�I�v�V������|�G�C���A�X|�Ӗ�|�f�t�H���g|
|:---:|:---:|:---:|:---:|
|--input_dir|-i|���̓f�B���N�g���̎w��|./images/|
|--save_image|-img|�o�̓f�B���N�g���̎w��|./output/|
|--save_log|-log|�����_���̃��O��csv�ɏo��|false|
|--method|-m|���o�A���S���Y���̎w��|0|
|--count|-c|count���[�h�ɕύX|false|

�Ȃ��C`-log`�R�}���h�͈����Ƃ��ďo�͂���郍�O�t�@�C���̖��O���w��ł���D

�܂��C`-m`�R�}���h�ł͒��o�A���S���Y����ԍ��Ŏw�肷��D�ԍ��ƑΉ�����A���S���Y���͈ȉ��̒ʂ�D

|�ԍ�|�A���S���Y��|
|:---:|:---:|
|0|ORB|
|1|AGAST|
|2|FAST|
|3|MSER|
|4|AKAZE|
|5|BRISK|
|6|KAZE|
|7|BLOB|
