from subprocess import call

url = r'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip'
name = './UCI HAR Dataset.zip'

cmd = 'wget -O%s%s' % (name, url)
call(cmd, shell=True)
