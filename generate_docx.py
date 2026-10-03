import pypandoc
import os

md_path = r'C:\Users\kotha\.gemini\antigravity\brain\30deb769-bb8c-46b8-8df0-3e50001f730a\Final_Report.md'
docx_path = r'C:\Users\kotha\Desktop\DL_project\Final_Report.docx'

def create_doc():
    try:
        print("Attempting to convert with pandoc...")
        pypandoc.convert_file(md_path, 'docx', outputfile=docx_path)
        print(f"Successfully created {docx_path}")
    except OSError:
        print("Pandoc not found. Downloading pandoc binary...")
        pypandoc.download_pandoc()
        print("Download complete. Converting...")
        pypandoc.convert_file(md_path, 'docx', outputfile=docx_path)
        print(f"Successfully created {docx_path}")

if __name__ == "__main__":
    create_doc()
