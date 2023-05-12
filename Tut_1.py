import cv2
import matplotlib.pyplot as plt

cb_img = cv2.imread('checkerboard_color.png')
coke_img = cv2.imread('coca-cola-logo.png')

#use matplotlib to show the image
plt.imshow(cb_img)
plt.title('matplot imshow')
plt.show()

#Use opencv to show the image
window1 = cv2.namedWindow('w1')
cv2.imshow(window1, cb_img)
cv2.waitKey(8000)
cv2.destroyWindow(window1)

#Use opencv to show the image
window2 = cv2.namedWindow('w2')
cv2.imshow(window2, coke_img)
cv2.waitKey(8000)
cv2.destroyWindow(window2)

#Use opencv imshow to show the image
window3 = cv2.namedWindow('w3')
cv2.imshow(window3, cb_img)
cv2.waitKey(0)
cv2.destroyWindow(window3)

window4 = cv2.namedWindow('w4')
Alive = True
while Alive:
    #Use opencv imshow to show the image until 'q' is pressed
    cv2.imshow(window4, coke_img)
    keypress = cv2.waitKey(1)
    if keypress == ord('q'):
        Alive = False
cv2.destroyWindow(window4)

cv2.destroyAllWindows()
stop=1
