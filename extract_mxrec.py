import mxnet as mx
import os

# Paths to your files
index_path = './data/train.idx'
data_path = './data/train.rec'

# Create an IndexedRecordIO object to read the data
record_io = mx.recordio.MXIndexedRecordIO(index_path, data_path, 'r')

img_info = record_io.read_idx(0)
header,_ = mx.recordio.unpack(img_info)
max_idx = int(header.label[0])
out_list = []
save_path = './output/'

for idx in range(1,  max_idx):
    img_info = record_io.read_idx(idx)
    header, img = mx.recordio.unpack(img_info)
    label = int(header.label)
    filename = "{}/{}_{}.jpg".format(label, label, idx)
    out_list.append("{} {}\n".format(filename, label))
    file_path = "{}/images/{}".format(save_path, label)
    if not os.path.isdir(file_path):
        os.makedirs(file_path)

    with open('{}/images/{}'.format(save_path, filename), 'wb') as file:
        file.write(img)
with open(os.path.join(save_path, "list.txt"), 'w') as f:
    f.writelines(out_list)