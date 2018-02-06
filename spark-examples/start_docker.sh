docker run -it --rm -p 8888:8888 jupyter/all-spark-notebook --mount source=data-vol,destination=/Users/eftim/Repositories/BigDataThinkTank/spark-examples/data
docker run -it --rm -p 8888:8888 jupyter/all-spark-notebook -v data-vol:/Users/eftim/Repositories/BigDataThinkTank/spark-examples/data
docker run -v /Users/eftim/Repositories/BigDataThinkTank/spark-examples/data:/data -it --rm -p 8888:8888 jupyter/all-spark-notebook
docker run -it --rm -p 8888:8888 jupyter/all-spark-notebook

docker cp /Users/eftim/Repositories/BigDataThinkTank/spark-examples/data jupyter/all-spark-notebook:/work