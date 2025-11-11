import chromadb

# bước 1: Tạo một client ChromaDB
# Đây là bước đầu tiên để kết nối với cơ sở dữ liệu ChromaDB.
# Dữ liệu sẽ được lưu trữ tạm thời trong bộ nhớ.
client = chromadb.Client()

# Để lưu trữ dữ liệu vĩnh viễn trên đĩa, bạn có thể sử dụng:
# client = chromadb.PersistentClient(path="./my_chromadb_directory")

print("ChromaDB client đã được tạo thành công.")

# bước 2: Tạo một collection (bộ sưu tập)
# Một cllection giống như một bảng trong MySQL.
# Nếu collection đã tồn tại, nó sẽ được tái sử dụng.
# Nếu chưa tồn tại, nó sẽ được tạo mới.
collection = client.get_or_create_collection(name="my_collection")

print("Collection đã được tạo hoặc lấy thành công.")

# bước 3: Thêm dữ liệu vào collection
# Đây là dữ liệu quan trọng nhất.
# ChromaDB sẽ tự động biến "documents" Thành "embeddings" sử dụng mô hình nhúng mặc định.
# Bằng một mô hình embedding mặc định
collection.add(
    documents=[ # đây là dữ liệu văn bản
        "Chroma DB là một cơ sở dữ liệu vector mã nguồn mở.",
        "Trí tuệ nhân tạo (AI) đang phát triển rất nhanh.",
        "Hà Nội là thủ đô của Việt Nam.",
        "Ốp la là một lựa chọn hợp lý cho buổi sáng.",
        "Mô hình ngôn ngữ lớn (LLM) cần rất nhiều dữ liệu để huấn luyện.",
        "Cơm tấm là món ăn phổ biến ở thành phố Hồ Chí Minh.",
    ],
    metadatas=[ # metadata là dữ liệu về dữ liệu
        {"source": "tech", "type": "database"},
        {"source": "tech", "type": "general-ai"},
        {"source": "geography", "type": "capital"},
        {"source": "food", "type": "personal"},
        {"source": "tech", "type": "llm"},
        {"source": "food", "type": "general"},
    ],
    ids=["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"],
)
print("Dữ liệu đã được thêm vào collection thành công.")

# bước 4: Tìm kiếm tương tự (similarity search)
# Đây là bước thể hiện sức mạnh của ChromaDB.
# chúng ta sẽ tìm kiếm dựa trên *ý nghĩa*, không phải từ khóa.

query_text = "Món ăn phổ biến ở Việt Nam"
print(f"Tìm kiếm tương tự cho câu hỏi: '{query_text}'")

#chúng ta sẽ truy vấn 2 kết quả tương tự nhất
results = collection.query(
    query_texts=[query_text],
    n_results=1
)

print("\n Kết quả tìm kiếm tương tự:")
print(results)

# bước 5: Truy vấn Nâng cao (lọc bằng Metadata)
# Bây giờ chúng ta sẽ thực hiện một truy vấn khác, nhưng chỉ lọc trong tài liệu 'tech'

#results_filtered = collection.query(
    #query_texts=[query_text],
    #n_results=2,
    #Đây là bộ lọc: chỉ tìm trong các tài liệu có metadata 'source' là 'tech'
    #where={"source": "tech"}
#)

#print("\n Kết quả tìm kiếm tương tự với bộ lọc (source=tech):")
#print(results_filtered)