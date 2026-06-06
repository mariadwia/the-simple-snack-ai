from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["https://thesimple.my.id"])

# Load data
df = None
try:
    df = pd.read_csv('data_snack_the_simple.csv')
    df.columns = df.columns.str.strip() 
    df['Harga'] = pd.to_numeric(df['Harga'], errors='coerce')
    df = df.dropna(subset=['Harga'])
    print("Database Berhasil Dimuat")
except Exception as e:
    print(f"Error Loading CSV: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if df is None:
        return jsonify({'status': 'error', 'message': 'Database tidak ditemukan'})

    try: 
        data = request.get_json()
        budget = int(data.get('budget', 0))
        people = int(data.get('people', 1))
        event = data.get('event', 'Arisan') # Menerima jenis acara

        # 1. Kalkulasi Budget
        target_per_box = budget / people
        fixed_cost = 2000 # Biaya Dus & Air
        available_for_snack = target_per_box - fixed_cost
        target_item_price = available_for_snack / 3

        if available_for_snack < 4500: 
            return jsonify({
                'status' : 'error', 
                'message' : f'Budget Rp {int(target_per_box)}/box terlalu rendah. Minimal Rp 6.500/box.'
            })
        
        # 2. LOGIKA MAPPING KATEGORI BERDASARKAN ACARA
        # Kategori dasar yang selalu muncul
        allowed_categories = ['Snack Asin', 'Snack Manis', 'Jajanan Pasar', 'Roti']

        # Jika acara Hantaran atau Arisan, munculkan Kue Kering & Kletikan
        if event in ['Hantaran', 'Arisan']:
            allowed_categories.append('Kue Kering')
            allowed_categories.append('Kletikan')
        
        # Jika Ulang Tahun, tambahkan opsi Bolu
        if event == 'Ulang Tahun':
            allowed_categories.append('Bolu')

        # Filter dataset berdasarkan aturan di atas
        snack_df = df[df['Kategori'].isin(allowed_categories)].copy()

        if len(snack_df) < 3:
            return jsonify({'status': 'error', 'message': 'Data tidak cukup untuk kombinasi ini.'})

        # 3. AI LOGIC (KNN)
        n_neighbors = min(10, len(snack_df))
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        knn.fit(snack_df[['Harga']].values)

        distances, indices = knn.kneighbors([[target_item_price]])
        
        # Stochastic Selection (Acak 3 dari 10 kandidat)
        n_candidates = indices[0]
        selected_indices = np.random.choice(n_candidates, min(3, len(n_candidates)), replace=False)

        recommendation_items = []
        total_snack_price = 0

        for i in selected_indices:
            row = snack_df.iloc[i]
            item = {
                'nama' : row['Produk'], 
                'harga' : int(row['Harga']), 
                'kategori': row['Kategori']
            }
            recommendation_items.append(item)
            
            # 4 LOGIKA BISNIS 
            total_snack_price = sum(item['harga'] for item in recommendation_items)

            # Selama total harga lebih mahal dari budget cemilan, DAN item masih lebih dari 1
            while total_snack_price > available_for_snack and len(recommendation_items) > 1:
                # Urutkan dari yang paling mahal di indeks ke-0
                recommendation_items = sorted(recommendation_items, key=lambda x: x['harga'], reverse=True)

                #Buang item termahal 
                recommendation_items.pop(0)

                #Hitung ulang
                total_snack_price = sum(item['harga'] for item in recommendation_items)

            # Pengecekan akhir jika tersisa 1 item tapi masih rugi 
            if total_snack_price > available_for_snack:
                return jsonify({
                    'status' : 'error', 
                    'message' : 'Sistem tidak dapat menemukan kombinasi produk yang sesuai dengan budget ini. Mohon naikkan budget sedikit yang lebih realistis!'
                })
            
        return jsonify({
            'status': 'success', 
            'calculation': {
                'target_per_box' : int(target_per_box), 
                'snack_budget_total' : int(available_for_snack),
                'real_snack_cost' : total_snack_price, 
                'remaining_margin' : int(available_for_snack - total_snack_price)
            }, 
            'recommendation' : recommendation_items, 
            'includes' : [
                {'item': 'Dus', 'price': 1000},
                {'item': 'Air Gelas', 'price' : 1000}
            ]
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Gagal memproses data server.'})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(port=5000, debug=True)