"""Clean PDF Parser with integrated product extraction"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

from config.config import INPUT_DIR, PARSED_JSON_DIR


class DocumentParser:
    """Clean PDF parser with proper product extraction - no messy line_items"""
    
    def __init__(self):
        self.input_dir = INPUT_DIR
        self.parsed_json_dir = PARSED_JSON_DIR
    
    def _extract_products(self, text: str) -> List[Dict[str, Any]]:
        """Extract products using multiple patterns"""
        products = []
        
        # Pattern 1: Product: NAME Quantity: X Unit Price: Y Total: Z
        pattern1 = r"Product:\s*([^\n]+?)\s+Quantity:\s*(\d+)\s+Unit\s*Price:\s*([\d.]+)\s+Total:\s*([\d.]+)"
        for match in re.finditer(pattern1, text, re.IGNORECASE):
            products.append({
                "product_name": match.group(1).strip(),
                "quantity": int(match.group(2)),
                "unit_price": float(match.group(3)),
                "total": float(match.group(4))
            })
        
        # Pattern 2: Table format (ID | Product | Qty | Price)
        if not products:
            pattern2 = r"(\d+)\s+([A-Za-z][^\d\n]+?)\s+(\d+)\s+([\d.]+)"
            for match in re.finditer(pattern2, text):
                try:
                    qty = int(match.group(3))
                    price = float(match.group(4))
                    products.append({
                        "product_id": match.group(1).strip(),
                        "product_name": match.group(2).strip(),
                        "quantity": qty,
                        "unit_price": price,
                        "total": qty * price
                    })
                except:
                    pass
        
        return products
    
    def _extract_metadata(self, text: str) -> Dict[str, str]:
        """Extract document metadata"""
        metadata = {}
        
        patterns = {
            "order_id": r"Order\s*ID:\s*(\d+)",
            "customer_id": r"Customer\s*ID:\s*([A-Z0-9]+)",
            "customer_name": r"Customer\s*Name:\s*([^\n]+)",
            "employee_name": r"Employee\s*Name:\s*([^\n]+)",
            "order_date": r"Order\s*Date:\s*([\d-]+)",
            "shipped_date": r"Shipped\s*Date:\s*([\d-]+)",
            "ship_name": r"Ship\s*Name:\s*([^\n]+)",
            "ship_city": r"Ship\s*City:\s*([^\n]+)",
            "ship_country": r"Ship\s*Country:\s*([^\n]+)",
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value and len(value) < 200:
                    metadata[key] = value
        
        return metadata
    
    def _detect_type(self, text: str, filename: str) -> str:
        """Detect document type"""
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        # Check filename first for more specific matching
        if "purchase_order" in filename_lower or "purchase order" in filename_lower:
            return "purchase_order"
        elif filename_lower.startswith("order_") or "/order_" in filename_lower or "\\order_" in filename_lower:
            return "order_doc"
        elif "invoice" in filename_lower or "inv-" in filename_lower or "inv_" in filename_lower:
            return "invoice"
        elif "grn" in filename_lower or "goods receipt" in filename_lower:
            return "grn"
        
        # Fallback to text content
        if "purchase order" in text_lower:
            return "purchase_order"
        elif "invoice" in text_lower:
            return "invoice"
        elif "grn" in text_lower or "goods receipt" in text_lower:
            return "grn"
        
        return "business_document"
    
    def parse_document(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """Parse PDF with clean product extraction"""
        if not PDFPLUMBER_AVAILABLE:
            print(f"   ❌ PDFPlumber not available")
            return None
        
        try:
            # Extract all text
            all_text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"
            
            # Extract products
            products = self._extract_products(all_text)
            
            # Extract metadata
            metadata = self._extract_metadata(all_text)
            
            # Detect type
            doc_type = self._detect_type(all_text, pdf_path.name)
            
            # Calculate total
            total_amount = sum(p.get('total', 0) for p in products)
            
            # Build clean document
            doc_data = {
                "document_id": metadata.get("order_id", pdf_path.stem),
                "source_file": pdf_path.name,
                "document_type": doc_type,
                "extraction_method": "pdfplumber_clean",
                "extraction_timestamp": datetime.now().isoformat(),
                "page_count": len(list(pdfplumber.open(pdf_path).pages)),
                "products": products,
                "product_count": len(products),
                "full_text": all_text,
                "text_length": len(all_text),
                **metadata
            }
            
            if total_amount > 0:
                doc_data["total_amount"] = total_amount
            
            return doc_data
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:100]}")
            return None
    
    def parse_all_documents(self, force_reparse: bool = False) -> List[Dict[str, Any]]:
        """Parse all PDFs in input folder"""
        pdf_files = list(self.input_dir.glob("*.pdf"))
        
        if not pdf_files:
            print("⚠️  No PDFs in input/")
            return []
        
        print(f"\n📚 Parsing {len(pdf_files)} document(s)\n")
        
        parsed_docs = []
        for pdf_path in pdf_files:
            json_path = self.parsed_json_dir / f"{pdf_path.stem}.json"
            
            if json_path.exists() and not force_reparse:
                print(f"✓ Cached: {pdf_path.name}")
                with open(json_path, 'r', encoding='utf-8') as f:
                    parsed_docs.append(json.load(f))
                continue
            
            print(f"📄 {pdf_path.name}")
            data = self.parse_document(pdf_path)
            
            if data:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                print(f"   ✅ {data['product_count']} products")
                if data.get('total_amount'):
                    print(f"   💰 ${data['total_amount']:.2f}")
                
                # Show first 3 products
                for i, prod in enumerate(data['products'][:3], 1):
                    name = prod.get('product_name', 'Unknown')[:40]
                    qty = prod.get('quantity', 0)
                    price = prod.get('unit_price', 0)
                    print(f"      {i}. {name} (Qty: {qty}, ${price})")
                
                parsed_docs.append(data)
        
        print(f"\n✅ Parsed {len(parsed_docs)}/{len(pdf_files)} documents\n")
        return parsed_docs
    
    def get_new_pdfs(self) -> List[Path]:
        """Get PDFs that haven't been parsed yet"""
        pdf_files = list(self.input_dir.glob("*.pdf"))
        return [p for p in pdf_files if not (self.parsed_json_dir / f"{p.stem}.json").exists()]
