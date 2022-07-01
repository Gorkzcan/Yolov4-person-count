class IzlenecekNesne:
     def __init__(self, nesneID, centroid):
		#nesneler için ID başlatılıyor
		#Geçerli centroid(merkez) kullanılıyor
        self.nesneID = nesneID
        self.centroids = [centroid]

		#hali hazırda sayılmış veya sayılmamış
		#nesne olup olmadığını belirtmek için kullanılan bir boolean fonksiyonu başlatıldı
        self.counted = False




        #Bu sınıfın amacı izlenen her bir nesne ile ilgili bilgileri depolamaktır.