import wget

url = 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar'

destination = 'ILSVRC2012_img_train.tar'

wget.download(url, destination)

url = 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar'

destination = 'ILSVRC2012_img_val.tar'

wget.download(url, destination)