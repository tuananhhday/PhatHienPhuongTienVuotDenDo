# Bài báo: Phát hiện phương tiện vượt đèn đỏ

Tài liệu chính là `main.tex`, viết theo mẫu ACL hai cột, khổ A4, cỡ chữ 11 pt.

Thông tin tác giả:

- Nguyễn Tuấn Anh
- Lớp 124231
- Không ghi mã sinh viên

## Biên dịch

Biên dịch bằng pdfLaTeX ba lượt để cập nhật trích dẫn, tham chiếu và mục lục:

```text
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Giữ nguyên thư mục `figures/` cạnh `main.tex`. Biểu đồ được tái tạo bằng:

```text
python generate_figures.py
```

Mục lục và Tuyên bố sử dụng AI nằm ở cuối tài liệu. Mức sử dụng AI được ghi là 70%.
