f = open('dataset_caltech256_train.txt','r')
filedata = f.read()
f.close()

newdata = filedata.replace("/home/users/saman/today/VGG19-TF","datasets")

f = open('dataset_caltech256_train.txt','w')
f.write(newdata)
f.close()
