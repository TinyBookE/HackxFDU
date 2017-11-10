import input_data
import model
import predict
music = input_data.read_data_sets('./music',one_hot = True)

#model.train_model(music)
#model.test(music)

a = predict.find(r"C:\Users\ReiEi\Desktop\FDU\Classfictor\music_t\jazz\Freddie King - Sweet Home Chicago.mp3")
print(a)
