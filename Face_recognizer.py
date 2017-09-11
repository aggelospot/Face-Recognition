import cv2 #Ta sxolia einai se greeklish giati me ellinika evgaze error o compiler


casc = "/home/mordai/tei/compvision/haarcascade_frontalface_default.xml" #to path tou xml arxeiou me ta classifiers gia thn anagnwrisi proswpou. Mprostini anagnwrisi (Frontal detection) mono.


faceCascade = cv2.CascadeClassifier(casc)


vidcapture = cv2.VideoCapture(0) #xrhsimopoioume thn default webcam tou upologisth

while (1):
    ret, frame = vidcapture.read()                  #diavazoume 1-1 ta frames apo tin camera kai ta apothikevoume stis metavlites return kai frame.

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #metatroph se grayscale gia na mporei na epeksergastei to frame h detectMultiscale

    faces = faceCascade.detectMultiScale(           #h detectMultiscale kanei tin anagnwrish prwsopou kai epistrefei ta apotelesmata sto faces
        gray,                                       # h grayscale eikona
        scaleFactor=1.1,                            # to scalefactor xrhsimopoieitai gia na ftiaxtei h 'pyramidwth klimaka' pou voithaei sto na ginetai anagnwrish prototupwn analoga to vathos. To 1.1 moy efere ta kalutera apotelesmata
        minNeighbors=6,                             #minneighbors - posous 'geitones' prepei na exei to ypopsifio orthogonio gia na diatirithei.
        minSize=(30, 30),                           #to elaxisto megethos tou orthogwniou pou xrhsimopoieitai gia ti prospelash ths eikonas
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Efoson egine h anagnwrisi, ftiaxnei orthogwnia sta shmeia pou anagnwristike to prwsopo
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

    # Display sthn othoni
    cv2.imshow('Face Detection', frame)

    #Otan o xrhsths pathsei "q" to programma kleinei
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vidcapture.release()
cv2.destroyAllWindows()