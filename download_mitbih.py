import wfdb
import os

def download_dataset():
    database_name = 'mitdb' 
    download_dir = os.path.join(os.getcwd(), 'mitdb_data')
    
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        
    print(f"Starting download to {download_dir}...")
    
    # CHANGED: pb_dir to db_dir
    wfdb.dl_database(
        db_dir=database_name, 
        dl_dir=download_dir, 
        records='all',       
        annotators='all',    
        keep_subdirs=True, 
        overwrite=False      
    )
    
    print("Download complete!")

if __name__ == "__main__":
    download_dataset()