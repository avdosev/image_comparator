import os, shutil
path = "images"
moveto = "dataset"

print('Ищем файлы')
files = []
for d in map(lambda d: os.path.join(path, d), os.listdir(path)):
    for f in os.listdir(d):
        files.append(os.path.join(d, f))
print('Переносим файлы')
for i, f in enumerate(files):
    src = f
    dst = os.path.join(moveto, f'{i}.jpg')
    shutil.move(src, dst)
