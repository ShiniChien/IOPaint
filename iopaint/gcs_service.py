import os
import io
import time
from typing import Optional, Union, BinaryIO
from google.cloud import storage
from google.oauth2 import service_account # pip install google-cloud-storage
from PIL import Image


class GoogleCloudStorageImageHandler:
    """
    Class để xử lý việc kết nối, xác thực và thao tác với Google Cloud Storage.
    Chuyên biệt cho việc xử lý file hình ảnh, chuyển đổi giữa file và PIL Image.
    """
    
    def __init__(
        self,
        credentials_path: str = "credentials/transwise-420713-d228ab810d17.json",
        bucket_name: str = "transwise-comics",
        project_id: Optional[str] = None
    ):
        # Kiểm tra xem file credentials tồn tại không
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"File credentials không tồn tại: {credentials_path}")
        
        # Tạo credentials từ file JSON
        self.credentials = service_account.Credentials.from_service_account_file(credentials_path)
        
        # Tạo client
        if project_id:
            self.client = storage.Client(project=project_id, credentials=self.credentials)
        else:
            self.client = storage.Client(credentials=self.credentials)
        
        self.bucket_name = bucket_name
        self.bucket = self.client.bucket(self.bucket_name)
        print(f"Đã kết nối thành công với Google Cloud Storage dưới project ID: {self.client.project}")
    
    def upload_image(
        self, source_file: Union[str, BinaryIO, Image.Image], 
        destination_blob_name: str
    ) -> str:
        """
        Upload một hình ảnh lên Google Cloud Storage.
        
        Args:
            source_file: Có thể là đường dẫn đến file, file object hoặc PIL Image.
            destination_blob_name (str): Tên của blob (file) trên GCS.
            
        Returns:
            str: ID của file đã upload (chính là blob name).
        """
        t1 = time.time()
        # Tạo blob
        blob = self.bucket.blob(destination_blob_name)
        
        # Xử lý upload dựa trên loại input
        if isinstance(source_file, str):
            # Nếu là đường dẫn file
            blob.upload_from_filename(source_file)
        elif isinstance(source_file, Image.Image):
            # Nếu là PIL Image
            img_byte_arr = io.BytesIO()
            # Lưu với format gốc hoặc mặc định là PNG
            img_format = source_file.format if source_file.format else 'PNG'
            source_file.save(img_byte_arr, format=img_format)
            img_byte_arr.seek(0)  # Trở về đầu stream
            blob.upload_from_file(img_byte_arr)
        elif isinstance(source_file, bytes):
            # Nếu là bytes
            blob.upload_from_string(source_file)
        else:
            # Nếu là file object
            blob.upload_from_file(source_file)
        
        time_taken = round(time.time() - t1, 3)
        print(f"Đã upload thành công file đến {destination_blob_name} trong {time_taken}s")
        return destination_blob_name, time_taken
    
    def download_image(
        self, blob_name: str, 
        destination_file: Optional[str] = None
    ) -> Image.Image:
        """
        Download một hình ảnh từ Google Cloud Storage và trả về dưới dạng PIL Image.
        
        Args:
            blob_name (str): Tên của blob (file) trên GCS.
            destination_file (str, optional): Nếu muốn lưu file về máy, cung cấp đường dẫn.
            
        Returns:
            PIL.Image.Image: Đối tượng PIL Image.
        """
        t1 = time.time()
        # Lấy blob
        blob = self.bucket.blob(blob_name)
        
        # Tạo bytes buffer
        img_bytes = io.BytesIO()
        
        # Download vào buffer
        blob.download_to_file(img_bytes)
        img_bytes.seek(0)  # Trở về đầu stream
        
        # Nếu cần lưu file về máy
        if destination_file:
            img = Image.open(img_bytes)
            img.save(destination_file)
            print(f"Đã lưu hình ảnh vào {destination_file}")
        
        time_taken = round(time.time() - t1, 3)
        print(f"Đã download thành công file {blob_name} trong {time_taken}s")
        return img_bytes.getvalue(), time_taken
    
    def get_image_by_id(self, file_id: str) -> Image.Image:
        """
        Lấy hình ảnh theo ID (blob_name) và trả về dạng PIL Image.
        
        Args:
            bucket_name (str): Tên của bucket.
            file_id (str): ID của file (chính là blob_name).
            
        Returns:
            PIL.Image.Image: Đối tượng PIL Image.
        """
        return self.download_image(file_id)
    
    def list_images(
        self, prefix: Optional[str] = None, 
        delimiter: Optional[str] = None
    ) -> list:
        """
        Liệt kê tất cả các file hình ảnh trong bucket.
        
        Args:
            prefix (str, optional): Tiền tố để lọc các blob.
            delimiter (str, optional): Ký tự phân cách để giả lập thư mục.
            
        Returns:
            list: Danh sách các blob name.
        """
        # Liệt kê blobs
        blobs = self.bucket.list_blobs(prefix=prefix, delimiter=delimiter)
        
        # Lọc ra các file hình ảnh phổ biến
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
        
        image_blobs = [
            blob.name for blob in blobs 
            if any(blob.name.lower().endswith(ext) for ext in image_extensions)
        ]
        
        return image_blobs
    
    def delete_image(self, blob_name: str) -> bool:
        """
        Xóa một file từ Google Cloud Storage.
        
        Args:
            blob_name (str): Tên của blob (file) trên GCS.
            
        Returns:
            bool: True nếu xóa thành công.
        """
        # Lấy blob
        blob = self.bucket.blob(blob_name)
        
        # Xóa blob
        blob.delete()
        
        print(f"Đã xóa thành công {blob_name}")
        return True

gcs_handler = GoogleCloudStorageImageHandler()

if __name__ == "__main__":
    # r = gcs_handler.list_images()
    r = gcs_handler.download_image("image.png")
    print(r)
