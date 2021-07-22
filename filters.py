import cv2
import numpy as np
import math

def getSize(person, chairObject, doc):
    heightPerson, widthPerson = person.shape[:2]
    heightChair, widthChair = chairObject.shape[:2]
    heightDoc, widthDoc = doc.shape[:2]
    print("Size of Images")
    print("Person(height, width): (",heightPerson,",",widthPerson,")","\n\n")
    print("Chair Object(height, width): (",heightChair,",",widthChair,")","\n\n")
    print("Document(height, width): (",heightDoc,",",widthDoc,")","\n\n")

def sharpen(person, chairObject, doc): 
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return [cv2.filter2D(person, -1, kernel), cv2.filter2D(chairObject, -1, kernel), cv2.filter2D(doc, -1, kernel)]


def emboss(person, chairObject, doc): 
    kernel = np.array([[0,-1,-1],[1,0,-1],[1,1,0]])
    return [cv2.filter2D(person, -1, kernel), cv2.filter2D(chairObject, -1, kernel), cv2.filter2D(doc, -1, kernel)]

def blur(person, chairObject, doc):
    return [cv2.GaussianBlur(person, (35, 35), 0), cv2.GaussianBlur(chairObject, (35, 35), 0), cv2.GaussianBlur(doc, (35, 35), 0)]


def grayScale(person, chairObject, doc):
    return [cv2.cvtColor(person, cv2.COLOR_BGR2GRAY), cv2.cvtColor(chairObject, cv2.COLOR_BGR2GRAY), cv2.cvtColor(doc, cv2.COLOR_BGR2GRAY)]

def negative(person, chairObject, doc):
    return [cv2.bitwise_not(person), cv2.bitwise_not(chairObject), cv2.bitwise_not(doc)]

def flipWarping(person, chairObject, doc):
    return [cv2.flip(person, 0), cv2.flip(chairObject, 0), cv2.flip(doc, 0)]

def mirrorWarping(person, chairObject, doc):
    return [cv2.flip(person, 1), cv2.flip(chairObject, 1), cv2.flip(doc, 1)]

def minFilter(person, chairObject, doc):
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, (11,11))
    return [cv2.erode(person, kernel), cv2.erode(chairObject, kernel), cv2.erode(doc, kernel)]

def maxFilter(person, chairObject, doc):
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, (11,11))
    return [cv2.dilate(person, kernel), cv2.dilate(chairObject, kernel), cv2.dilate(doc, kernel)]


def applyConcaveEffect(image):
    rows, columns = image.shape[:2]
    imageResult = np.zeros(image.shape, dtype=image.dtype)
    for i in range(rows): 
        for j in range(columns): 
            moveX = int(128.0 * math.sin(2 * 3.14 * i / (2*columns))) 
            moveY = 0 
            if j+moveX < columns: 
                imageResult[i,j] = image[i,(j+moveX)%columns] 
            else: 
                imageResult[i,j] = 0 

    return imageResult

def concaveEffect(person, chairObject, doc):
    return [applyConcaveEffect(person), applyConcaveEffect(chairObject), applyConcaveEffect(doc)]


def allFilters(person, chairObject, doc):
    sharpens = sharpen(person, chairObject, doc) # realça a imagem
    embossing = emboss(person, chairObject, doc) #destaca o relevo, no caso as linhas
    gaussianBlur = blur(person, chairObject, doc) # imagem borrada, filtro linear
    grayScales = grayScale(person, chairObject, doc) #filtro de escala cinza
    negatives = negative(person, chairObject, doc) # filtro de imagem negativa, filtro estatistico e de amplitude
    flipWarpingImg = flipWarping(person, chairObject, doc) # filtro que inverte a image, filtro deterministico
    mirrorWarpingImg = mirrorWarping(person, chairObject, doc) #filtro que espelha a iamge, filtro deterministico
    minimumFilter = minFilter(person, chairObject, doc) # filtro topologico minumo, menor itensidade de rgb de um pixel
    maximumFilter = maxFilter(person, chairObject, doc) # filtro topologico maximo, maior internsidade de rgb de um pixel
    concaveFilterEffect = concaveEffect(person, chairObject, doc) # filtro de aplicar efeito concavo, quase um warping...

    return [sharpens, embossing, gaussianBlur, grayScales, negatives, flipWarpingImg, mirrorWarpingImg, minimumFilter, maximumFilter, concaveFilterEffect]



def showFilters():
    person = cv2.imread('me.jpeg', 1)
    chairObject = cv2.imread('chair.jpeg', 1)
    doc = cv2.imread('doc.jpeg', 1)

    getSize(person, chairObject, doc)
    filters = allFilters(person, chairObject, doc)
    
    nameOfFilters= ["Sharpen", "Embossing", "GaussBlur", "GrayScale", "Negative", "FlipWarping", "MirrorWarping", "MinimumFilter", "MaximumFilter", "ConcaveFilterEffect"]
    nameOfImages = ["person", "chair", "doc"]
    for i in range(len(filters)):
        imageString = ""
        for j in range(len(filters[i])):
            imageString = "/home/dev03/Eng.Lucas/FifthProject-CG-PDI/"+nameOfImages[j]+nameOfFilters[i]+str(i)+"-"+str(j)+".png"
            cv2.imshow(imageString, filters[i][j])
            cv2.imwrite(imageString, filters[i][j])
            cv2.destroyAllWindows()
    


showFilters()