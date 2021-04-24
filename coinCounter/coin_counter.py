import cv2 as cv

img = cv.imread("coin_test.jpeg")

height, width = img.shape[:2]
res = cv.resize(img,(width/2, height/2), interpolation = cv.INTER_LINEAR)

grey = cv.cvtColor(res, cv.COLOR_BGR2GRAY)

blurred = cv.GaussianBlur(grey, (17, 17), 0)

outline = cv.Canny(blurred, 30, 150)
(cnts, _) = cv.findContours(outline, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

res = cv.drawContours(res, cnts, -1, (0, 255, 0), 2)
res = cv.putText(res, "Jumlah koin: %i " % len(cnts), (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv.LINE_AA)
cv.imwrite('hasildeteksi.jpeg', res)
cv.imshow("Final Results", res)

cv.waitKey(0)
cv.destroyAllWindows()