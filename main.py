from mpi4py import MPI
import numpy as np
import sys

def main():
    # MPI başlatma
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = 0
    matrix_a = None
    matrix_b = None

    # 1. DOSYADAN OKUMA ( Master İşlemci - Rank 0)
    if rank == 0:
        try:
            # a.txt okuma
            with open('a-v2.txt', 'r') as f:
                lines = f.readlines()
                N = int(lines[0].strip())
                matrix_a = np.array([list(map(float, line.split())) for line in lines[1:N+1]], dtype=np.float64)
            
            # b.txt okuma
            with open('b-v2.txt', 'r') as f:
                lines = f.readlines()
                N_B = int(lines[0].strip())
                matrix_b = np.array([list(map(float, line.split())) for line in lines[1:N_B+1]], dtype=np.float64)
            
            if N != N_B:
                print("Hata: A ve B matrislerinin boyutları eşleşmiyor!")
                comm.Abort()
                
        except Exception as e:
            print(f"Dosya okuma hatası: {e}")
            comm.Abort()

    # N değerini tüm işlemcilere yayınla (Broadcast)
    N = comm.bcast(N, root=0)

    # 2. B MATRİSİNİN YAYINLANMASI (Bcast)
    if rank != 0:
        matrix_b = np.empty((N, N), dtype=np.float64)
    comm.Bcast(matrix_b, root=0)

    # 3. A MATRİSİNİN DAĞITILMASI (Scatter)
    if rank == 0:
        # A matrisini işlemci sayısına göre satır satır böl
        split_a = np.array_split(matrix_a, size, axis=0)
    else:
        split_a = None

    # Her işlemci kendi payına düşen A satırlarını alır
    local_a = comm.scatter(split_a, root=0)

    # Süre ölçümünden önce tüm işlemcilerin aynı noktaya gelmesini bekle
    comm.Barrier()
    start_time = MPI.Wtime()

    # 4. HESAPLAMA (Yerel Çarpım)
    # Kendi payına düşen A satırları ile B'nin tamamını çarp
    local_c = np.dot(local_a, matrix_b)

    # Süre ölçümünü bitirmeden önce tüm işlemlerin bitmesini bekle
    comm.Barrier()
    end_time = MPI.Wtime()

    # 5. SONUÇLARIN TOPLANMASI (Gather)
    gathered_c = comm.gather(local_c, root=0)

    # 6. RAPORLAMA (Master İşlemci)
    if rank == 0:
        # Toplanan parçaları dikey olarak birleştirerek nihai C matrisini oluştur
        matrix_c = np.vstack(gathered_c)
        parallel_time = end_time - start_time
        
        print(f"--- MPI Python Performans Raporu ---")
        print(f"İşlemci Sayısı (P): {size}")
        print(f"Matris Boyutu (N): {N}x{N}")
        print(f"Paralel Süre (T_P): {parallel_time:.6f} saniye")
        
        if N <= 10:
            print("\nHesaplanan C Matrisi:")
            print(matrix_c)

if __name__ == '__main__':
    main()
