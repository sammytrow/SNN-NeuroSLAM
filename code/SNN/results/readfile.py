import pickle

# read python dict back from the file
pkl_file = open('input_landmark_fov_x.pkl', 'rb')
mydict2 = pickle.load(pkl_file)
pkl_file.close()

print (mydict2)
pkl_file = open('input_landmark_fov_y.pkl', 'rb')
mydict2 = pickle.load(pkl_file)
pkl_file.close()

print (mydict2)
pkl_file = open('input_pos_n_velocity.pkl', 'rb')
mydict2 = pickle.load(pkl_file)
pkl_file.close()

print (mydict2)

pkl_file = open('output_y_pos.pkl', 'rb')
mydict2 = pickle.load(pkl_file)
pkl_file.close()

print (mydict2)
pkl_file = open('output_x_pos.pkl', 'rb')
mydict2 = pickle.load(pkl_file)
pkl_file.close()

print (mydict2)
pkl_file = open('output_theta.pkl', 'rb')
mydict2 = pickle.load(pkl_file)
pkl_file.close()

print (mydict2)