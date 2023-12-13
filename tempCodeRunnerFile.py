f = open(os.getcwd() + "/data_by_part.json", 'wb')
pickle.dump(data_by_part, f)
f.close()
