import cv2
import numpy as np

cam=cv2.VideoCapture(0)
#ask your name
filename=input("Enter your name: ")
dataset_path = "./data/"
offset = 20

#model
model=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

facedata=[]
skip = 0

while True:
  success,img=cam.read()
  if not success:
    print("Your camera is not working")

  #store the gray images
  grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  faces=model.detectMultiScale(img,1.3,5)
  faces=sorted(faces,key=lambda f:f[2]*f[3])

  if(len(faces)>0):
    f=faces[-1]

    x,y,w,h=f
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
    
    cropped_face=img[y-offset:y+h+offset,x-offset:x+w+offset]
    cropped_face=cv2.resize(cropped_face,(100,100))
    skip+=1
    if skip%10==0:
      facedata.append(cropped_face)
      print("Saved so far= "+str(len(facedata)))
  cv2.imshow("AKM",img)
  # cv2.imshow("lAKM",cropped_face)
  key=cv2.waitKey(1)
  if key==ord('q'):
    break
  
facedata = np.asarray(facedata)
m=facedata.shape[0]
facedata=facedata.reshape((m,-1))

print(facedata.shape)

#save this data into file system
filepath=dataset_path+filename +".npy"
np.save(filepath,facedata)
print("Data successfully saved at "+filepath)
  

cam.release()
cv2.destroyAllWindows()