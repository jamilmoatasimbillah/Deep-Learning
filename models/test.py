import cv2
import time

def predict_on_batch(model, imgs):
    targets = [cv2.resize(img, (512, 512), cv2.INTER_AREA) for img in imgs]
    frames = [cv2.resize(img, (128, 128)) for img in targets]
    frames = [cv2.resize(img, (512, 512), cv2.INTER_CUBIC) for img in frames]
    
    predicted = model.predict_on_batch(keras.backend.variable(frames))
    return targets, frames, predicted



def validate(inputs, outputs):
    pass

def test( imgs, save_dir=None):
    targets, frames, predicted = predict(generator, test_imgs)  
    
    if save_dir:
        for j in range(len(frames)):
          img = np.zeros(shape=(512, 512*3, 3))
          img[:, :512, :] = targets[j]
          img[:,512:1024, :] = cv2.resize(frames[j], (512,512), cv2.INTER_CUBIC)
          img[:,1024:, :] = predicted[j]
          img[:,1024:, :][img[:,1024:, :]>1] = 1
          img[:,1024:, :][img[:,1024:, :]<0] = 0
          plt.imsave(f"{save_dir}/{j+1}_{time.ctime()}.png",img)