from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

#maxKaybolma -> Bir nesnenin kaydı silinene kadar "kayıp/kayboldu" olarak işaretlenmesine izin verilecek ardışık frame sayısı
class CentroidIzle:
	def __init__(self, maxKaybolma=50, maxMesafe=50):
		"""
		sıralı iki nesneyle birlikte bir sonraki benzersiz nesne kimliği başlatılıyor
		belirlenen nesneyi takip edebilmek için Dictionary oluşturuluyor
		Merkezin kimliği ve sahip olduğu ardışık çerçeve sayısı sırasıyla işaretleniyor
		"""
		self.nextNesneID = 0
		self.nesneler = OrderedDict()
		self.kaybolma = OrderedDict()
		
		"""
		verilen maksimum ardışık çerçeve sayısı depolanır
		nesnenin kaybolana kadar izleme kaydı tutuluyor daha sonrasında
		bu kayıt siliniyor.
		"""
		self.maxKaybolma = maxKaybolma

		"""
		ilişkilendirmek için merkezler arasındaki maksimum mesafe ayarlanıyor.
		Takip edilen bir nesnenin mesafesi eğer maksimum olarak tanımlanan değerden büyükse
		Kaybolma eşiğine geldiğinde nesne kayboldu olarak tanımlanır
		"""
		self.maxMesafe = maxMesafe

	def kayit(self, centroid):
		"""
		bir nesne kaydedilirken bir sonraki kullanılabilir nesneyi 
		izlemeye başlarız
		Merkezleri depolayabilmek için kimlik ataması yapılır
		"""
		self.nesneler[self.nextNesneID] = centroid
		self.kaybolma[self.nextNesneID] = 0
		self.nextNesneID += 1

	def kayitSil(self, nesneID):
		#nesne kimliğiniin kaydını silip 'ID''i boşa çıkartmak için nesne kimliği siliniyor
		del self.nesneler[nesneID]
		del self.kaybolma[nesneID]

	def guncelle(self, rects):
		"""
		nesnelere çizilen sınırlayıcı kutuların listesinin boş olup olmadığını denetliyoruz
		"""
		if len(rects) == 0:
			# mevcutta izlenen bir nesne varsa ve bu nesne kaybolursa döngü sayesinde nesneyi kayboldu olarak işaretledik
			for nesneID in list(self.kaybolma.keys()):
				self.kaybolma[nesneID] += 1

				#belirlenen nesnenin kayboldu olarak işaretlendiği maksimum ardışık çerçeve sayısına ulaşıldıysa kaydı silinsin
				if self.kaybolma[nesneID] > self.maxKaybolma:
					self.kayitSil(nesneID)

			"""
			güncellenecek bir merkez veya izleme bilgisi yoksa geri dön
			"""
			return self.nesneler

		# Anlık frame(çerçeve) için bir merkez girişi dizisi başlatılyıor
		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		# sınırlayıcı kutular üzerinde döngüye giriliyor
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			# Centroid(merkez) atayabilmek için sınırlayıcı kutu koordinatlarının verisi çekiliyor
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		#anlık olarak herhangi bir nesne izlenmiyorsa girdi alınır ve her biri kaydedilir.
		if len(self.nesneler) == 0:
			for i in range(0, len(inputCentroids)):
				self.kayit(inputCentroids[i])

		#aksi takdirde, anlık izlenilen nesne varsa bu nesnenin merkezini mevcut
		# nesneye ata
		else:
			# nesne ID'leri ve karşılık gelecek merkezlerinin kümesi alınır
			nesneIDs = list(self.nesneler.keys())
			nesneCentroids = list(self.nesneler.values())

			#her bir nesne merkezi ile giriş yapılacak alan arasındaki merkez mesafesi hesaplanır
			#burada amaç bir giriş merkezini mevcut nesne ile eşleyebilmek olacaktır.
			D = dist.cdist(np.array(nesneCentroids), inputCentroids)

			#Eşleyebilmek için her satırdaki en küçük değer bulunur ve satır dizinleri
			# minimum değere göre sıralanır
			
			rows = D.min(axis=1).argsort()

			#Amaçlanan karşılık gelen en küçük mesafeye sahip indeks değerlerinin listelerin
			#başında yer almasıdır.
			cols = D.argmin(axis=1)[rows]

			#hangi satır ve sütunu kullandığımızı belirlemek için iki küme başlatıyoruz
			#Bu kümeler bir listeye benziyor ama benzersiz değerler içeriyor çünkü her nesne için farklı değerler üretiliyor
			usedRows = set()
			usedCols = set()

			#ardından, nesne merkezlerimizi güncellemek için satır, sütun gruplarının içerisinde dolaşıyoruz.
			for (row, col) in zip(rows, cols):
				#içerisinde gezindiğimiz satır veya sütun dizisi zaten kullanıldıysa onu yok sayıp döngüye devam ediyoruz
				if row in usedRows or col in usedCols:
					continue

				#merkezler arası mesafe maksimum mesafeden (50) büyükse, iki merkez aynı nesneyle ilişkinlendirilmez döngüye devam edilir
				if D[row, col] > self.maxMesafe:
					continue

				#yukarıdakilerin aksine varolan merkeze en küçük öklid mesafesine sahip ve başka hiç bir nesne ile eşleşmemiş bir girdi bulunursa nesne merkezi güncelleniyor.
				nesneID = nesneIDs[row]
				self.nesneler[nesneID] = inputCentroids[col]
				self.kaybolma[nesneID] = 0

				#sırasıyla tüm satır ve sütunlar incelendi(ekleme başarılımı diye)
				usedRows.add(row)
				usedCols.add(col)

			#hala incelenmemiş satır ve sütun kalmışsa, hangi merkez indeksleri henüz incelenmediği belirlenir ve bunları unusedRows, unusedCols değişkenlerine atarız.
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			#Son kontrolümüz, kaybolan veya kaybolma olasılığı bulunan tüm nesneleri ele alır:
			#nesnenin merkezi değeri, giriş kısmının merkezi değerinden büyük veya eşitse:
			if D.shape[0] >= D.shape[1]:
				# kullanılmayan satır dizinleri üzerinde döngüye girilir
				for row in unusedRows:
					#karşılık gelen satırlar için nesne kimliği alınır
					#kaybolanları tutan sayaçların indeks değerleri 1 arttırılır
					nesneID = nesneIDs[row]
					self.kaybolma[nesneID] += 1

					#Kaybolanların sayısının maxKaybolma eşiğini (50)
					#aşıp aşmadığını kontrol ediyoruz ve eğer öyleyse nesnenin kaydını sileceğiz

					if self.kaybolma[nesneID] > self.maxKaybolma:
						self.kayitSil(nesneID)

			#Giriş kısmının merkezleri sayısı mevcut nesne merkezleri sayısından daha fazlaysa yeni giriş yapan nesneler mevcuttur
			#Bu nedenle kaydedilmeye hazır yeni nesneler olacaktır
			
			else:
				for col in unusedCols:
					self.kayit(inputCentroids[col])

		# izlenebilir nesneler dizisini geri döndürüyoruz.
		return self.nesneler