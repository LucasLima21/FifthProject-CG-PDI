import cv2

person = cv2.imread('me.jpeg', 1)
print(person)

cv2.imshow('Person', person)
cv2.waitKey(5000)
cv2.destroyWindows()