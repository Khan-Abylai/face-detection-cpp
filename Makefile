image=cpp_ncnn
tag=v1.0
tag_dev=dev
dockerfile = ${image}:&{tag}
build:
	docker build -t dockerfiles/${image}:${tag} -f Dockerfile .

build-dev:
	docker build -f ${dockerfile} -t ${image}:${teg_dev} .

run:
	docker run -d -it --rm --ipc=host --cap-add sys_ptrace -p127.0.0.1:3333:22 -v /home/yeleussinova/data_SSD/plates/:/plates/ \
                --gpus all \
                --name lp_train \
               cpp_ncnn:1.0