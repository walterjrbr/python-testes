import cv2
import sys

# Get user supplied values
#imagePath = sys.argv[1]
imagePath = "onde está sua imagem"
cascPath = "arquivo haar cascade xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
# Aqui os valores podem ser ajustados para melhor desempenho
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.5,
    minNeighbors=1,
    minSize=(50, 50),
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    flags=cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
