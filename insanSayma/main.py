import cv2 
import numpy as np
from centroidIzle import CentroidIzle
from IzlenebilirNesne import IzlenecekNesne

# Parametreleri Tanımladık
confThreshold = 0.6  #Güven Esigi
nmsThreshold = 0.4   #Maksimum olmayanı bastırma eşiği
inpWidth = 416       #Kaynak görüntünün genişliği
inpHeight = 416      #Kaynak görüntünün uzunluğu
"""
cfg dosyasında 416,416 boyutlarının yolov4 için daha iyi olduğu yazıyor. Bu sebeple boyut 416,416 seçildi
"""
cap = cv2.VideoCapture('test1.mp4')
        
# Coconames sınıflarını yükledik
classesFile = "coco.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Yolov4'e dair config ve weights dosyalarını atamak için değişken yarattık.
modelConfiguration = "yolov4-tiny.cfg";
modelWeights = "yolov4-tiny.weights";

# modelimizi diskimizden sisteme yükledik

net = cv2.dnn.readNet(modelConfiguration, modelWeights)
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# merkez izleyicimizi somutlaştırdık, ardından depolamak için bir liste başlattık
# yakalanan her bir  nesne kimliğini bir IzlenebilirNesne ile eşledik
ct = CentroidIzle(maxKaybolma=50, maxMesafe=50)
trackers = []
izlenebilirNesneler = {}

 
#saymak istediğimiz nesnelerin yukarı veya aşağı hareketlerini saymak için değişkenleri tanımladık
toplamAsagi = 0
toplamYukari = 0


def katmanAd(net):
    # Ağın tüm katmanlarının adını aldık
    layersNames = net.getLayerNames()
    # getUnconnectedOutLayer(): -> çıktı katmanlarının dizinini alıyoruz.
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]
    

# YOLO, evrişimli özellik çıkarma üzerine tamamen bağlı katmanları kullanarak doğrudan sınırlayıcı kutuların koordinatlarını tahmin eder.
#SinirKutu fonksiyonu bu kutunun koordinatlarını bulabilmek için yazılmıştır.

def SinirKutu(classId, conf, left, top, right, bottom):
    # Sınırlayıcı kutu çizimi
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 2)
    # sınırlayıcı kutunun merkezini belirlemek
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    #yöne göre sayım yapabilmek için sayac değişkenini oluşturuyoruz. 
    sayac = 0
    if (top+(bottom-top)//2 in range(frameHeight//2 - 2,frameHeight//2 + 2)):
        sayac +=1

# Düşük güven skoruna sahip olan YOLO tarafından oluşturulmuş diğer box'ların kaldırılmasının fonksiyonu
def nmsFonk(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    rects = []

    #Burada yapılan nmsThreshold işlemidir. Yani ağdan geri dönen tüm sınırlayıcı kutular taranır ve yalnızca yüksek güven puanına sahip olanlar
    # alınır. Kutunun sınıf etiketi(person, car vb.) en yüksek puana sahip sınıf olarak atanır.
    #Modele her bir frame direkt gönderiliyor. Buna göre model bize bir çıktı veriyor. Burada ilk 4 eleman kutu koordinatlarıdır. Bu çıktılar 0-1 arasında olduğu için
    #frame boyutu ile çarpıyoruz. Ayrıca buradaki ilk iki değer nesnenin ortası fakat bize kutuyu çizebilmek için sol üst köşe lazım left ve top sol üst köşeye tekabul edecek.
    #Scores her bir sınıf için bulunan değerdir. Biz bunun en yüksek olanını alıyoruz eğer bu değer belirlediiğimiz değerden yüksekse işlemleri buna göre yapıyoruz.

    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            #detection[5:] -> ilk 4 öge center_x, center_y, genişlik ve yüksekliktir. Son öğe ise sınırlayıcı kutuların nesneyi çevrelediğindeki güven faktörüdür.
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                

    #üst üste binen ve güven eşiği daha düşük olan, nesneyi tam olarak çerçevelemeyen tüm kutuları ortadan kaldırabilmek için maksimum olmayını bastırma işlemi (nmsThreshold)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        #her kutu geziliyor
        # Class "person"
        if classIds[i] == 0:
            #classIds person classına eşitse gezilen kutular değerlerine göre tersine sıralanır(en yüksekten en düşüğe) ve bunun sonucunda en yüksek güven skorlu class elde tutulur.
            rects.append((left, top, left + width, top + height))
            confidences.sort(reverse=True)
            conf = str(confidences[0]*100)
            #izleme algoritmasına göre tanımlanmış en yüksek güven skorlu nesneyi ilişkilendirmek için centroid izleyicisini kullanıyoruz
            nesneler = ct.guncelle(rects)
            sayici(nesneler)
            cv2.putText(frame, conf, (left+width//4,top+height//4 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            SinirKutu(classIds[i], confidences[i], left, top, left + width, top + height)

def sayici(nesneler):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    global toplamAsagi
    global toplamYukari

    # izlenen nesnelerin döngüsü
    for (nesneID, centroid) in nesneler.items():
        #halihazırda izlenen bir nesne varsa bunun için yeni bir merkez yaratmıyoruz
        to = izlenebilirNesneler.get(nesneID, None)
 
        # eğer izlenen herhangi bir öğe yoksa ilk izlenecek öğeye yeni bir merkez atanıyor
        if to is None:
            to = IzlenecekNesne(nesneID, centroid)
 
        #eğer izlenene bir nesne varsa biz aynı zamanda yönünü belirlemek için kullanabiliriz(Takip algoritmasını)
        else:
            """
            Mevcut merkezin y koordinatı ile önceki merkezin y koordinatı arasındaki fark bize nesnenin hangi yönde hareket ettiğini söyleyecek.
            yukarı için negatif aşağı için pozitif(şimdiki - bir önceki)
            """
            y = [c[1] for c in to.centroids]
            yonBilgi = centroid[1] - np.mean(y)
            to.centroids.append(centroid)
            #print(centroid[1], centroid[0])
            # nesnenin sayılıp sayılmadığını kontrol ediyoruz
            if not to.counted:
                #yön negatifse nesnenin yukarıya gittiğini gösterecektir.. Ve merkez orta çizginin üzerindeyse nesne sayılacaktır.

                if yonBilgi < 0 and centroid[1] in range(frameHeight//2 - 30, frameHeight//2 + 30):
                    toplamYukari += 1
                    to.counted = True
                    
 
                #yön pozitifse nesnenin aşağıya gittiğini gösterecektir.. Ve merkez orta çizginin üzerindeyse nesne sayılacaktır.
                elif yonBilgi > 0 and centroid[1] in range(frameHeight//2 - 30, frameHeight//2 + 30):
                    toplamAsagi += 1
                    to.counted = True
 
        # izlenilen nesne dictionary'de saklanıyor.
        izlenebilirNesneler[nesneID] = to
        #nesnenin merkezine çıktı ID'si eklenecek
        text = "ID {}".format(nesneID)
        cv2.putText(frame, text, (centroid[0] - 15, centroid[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)
    #frame içerisinde görüntülemek için bir dizi oluşturup içine bilgi yazdiriyoruz.
    bilgi = [
        ("Yukari", toplamYukari),
        ("Asagi", toplamAsagi),
    ]

    # yazdırdığımız dizinin içerisinde dolaşıp bunu frame'e çizdiriyoruz.
    for (i, (k, v)) in enumerate(bilgi):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, frameHeight - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)



while cv2.waitKey(1) < 0:
    
    # videodan çerçeve çekiyoruz
    ret, frame = cap.read()
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    #print(frame.shape) videonun boyutları (720, 1280, 3)   
    cv2.line(frame, (0, frameHeight // 2), (frameWidth, frameHeight // 2), (0, 255, 255), 2)
    #frame1 = cv2.resize(frame, (416,416))
    #cv2.imshow("Frame1", frame1)
    #
    cv2.line(frame, (0, frameHeight//2-30), (frameWidth, frameHeight//2-30), (255,255,255), 2)
    cv2.line(frame, (0, frameHeight//2+30), (frameWidth, frameHeight//2+30), (255,255,255), 2)
    

    #Tahmin edilen nesne konumlarını elde edebilmek için frame'i sinir ağları nesne dedektörüne verdik
    blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # ayarladığımız blobu tanımladığımız net'e veriyoruz
    net.setInput(blob)

    # çıktı katmanlarının verisini döndürebilmek için tanımlama yaptık
    outs = net.forward(katmanAd(net))
    

    # düşük güven skorlu kutuları kaldırmak için frame ve çıkış fonksiyona iletildi
    nmsFonk(frame, outs)

    cv2.imshow("Frame",frame)