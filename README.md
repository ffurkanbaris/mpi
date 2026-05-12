#  MPI Parallel Matrix Multiplication Analysis

Bu proje, **Message Passing Interface (MPI)** kullanılarak hem düşük seviyeli (C - OpenMPI) hem de yüksek seviyeli (Python - mpi4py) dillerde geliştirilmiş paralel matris-matris çarpımı (C = A x B) uygulamalarını içermektedir. 

Çalışmanın temel amacı, donanım kaynaklarının (çekirdek sayısı) paralel performansa etkisini ölçmek; oversubscription (çekirdek aşımı) durumlarında işletim sisteminin bağlam değişimi (context switching) maliyetlerini gözlemlemektir.

##  Özellikler ve Mimari
* **Master-Worker Paradigması:** `Rank 0` (Master) süreci verileri okur ve sonuçları toplar. İşçi (Worker) süreçler matematiksel hesaplamaları gerçekleştirir.
* **Veri Dağıtımı (MPI_Scatter):** `A` matrisi satır bloklarına bölünerek tüm süreçlere eşit olarak dağıtılır.
* **Veri Yayını (MPI_Bcast):** `B` matrisinin tamamı iletişim maliyeti gözetilerek tüm süreçlere kopyalanır.
* **Sonuç Toplama (MPI_Gather):** İşçi süreçlerde hesaplanan lokal `C` matrisi parçaları Master süreçte birleştirilir.
* **Dinamik Boyutlandırma:** Algoritma, matris boyutunu (N) dosyanın ilk satırından dinamik olarak okuyacak şekilde tasarlanmıştır.
* **Not:** a.txt ve b.txt dosyaları, 8x8 float veri tipinde matrislerden, a-v2 ve b-v2 ise 512x512 float veri tipinde matrislerden oluşmaktadır.

##  Kullanılan Teknolojiler
* **Diller:** C, Python 3
* **Kütüphaneler:** `<mpi.h>`, `mpi4py`, `numpy`
* **Ortam:** Windows Subsystem for Linux (WSL2 - Ubuntu)

##  Kurulum ve Gereksinimler

Projeyi yerel ortamınızda çalıştırmak için OpenMPI geliştirme araçlarına ve Python kütüphanelerine ihtiyacınız vardır. Ubuntu/Debian tabanlı sistemlerde kurulum için:

```bash
# OpenMPI kütüphanelerini kurun
sudo apt update
sudo apt install openmpi-bin libopenmpi-dev

# Python MPI ve Numpy kütüphanelerini kurun
sudo apt install python3-mpi4py python3-numpy

 Çalıştırma
Öncelikle a.txt ve b.txt dosyalarının ana dizinde bulunduğundan emin olun. (Dosyaların ilk satırı matris boyutunu (N), devamı ise matris verilerini içermelidir).

C (OpenMPI) İçin:

# Kodu derleyin (-O3 bayrağı ile optimizasyon yapılabilir)
mpicc main.c -o mpi_c

# 8 süreç (process) ile çalıştırın
mpirun -np 8 ./mpi_c
Python (mpi4py) İçin:

# 8 süreç ile çalıştırın
mpirun -np 8 python3 main.py

# Not: Fiziksel işlemci sayınızdan daha fazla süreç başlatmak isterseniz, komutunuza --oversubscribe bayrağını eklemeniz gerekmektedir (Örn: mpirun --oversubscribe -np 16 ./mpi_c).
