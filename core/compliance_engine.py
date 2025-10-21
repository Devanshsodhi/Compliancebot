"""
Compliance Reasoning Engine
Orchestrates document analysis, rule processing, and compliance reporting
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json

from core.llm_orchestrator import LLMOrchestrator, PromptTemplates
from core.document_parser import DocumentParser
from core.vector_store import DocumentVectorStore
from config.config import COMPLIANCE_RULES_FILE, COMPLIANCE_REPORT_TEMPLATE


class ComplianceEngine:
    """
    Core compliance reasoning engine
    Combines document intelligence, rules, and LLM reasoning
    """
    
    def __init__(self):
        self.llm = LLMOrchestrator()
        self.parser = DocumentParser()
        self.vector_store = DocumentVectorStore()
        self.compliance_rules = self._load_compliance_rules()
        
        print("⚖️  Compliance Engine initialized")
    
    def _load_compliance_rules(self) -> str:
        """
        Load compliance rules from text file
        Rules are written in natural language for LLM processing
        """
        if not COMPLIANCE_RULES_FILE.exists():
            # Create default rules file
            default_rules = self._get_default_rules()
            with open(COMPLIANCE_RULES_FILE, 'w', encoding='utf-8') as f:
                f.write(default_rules)
            print(f"✅ Created default compliance rules: {COMPLIANCE_RULES_FILE.name}")
            return default_rules
        
        with open(COMPLIANCE_RULES_FILE, 'r', encoding='utf-8') as f:
            rules = f.read()
        
        print(f"✅ Loaded compliance rules from {COMPLIANCE_RULES_FILE.name}")
        return rules
    
    def _get_default_rules(self) -> str:
        """
        Default compliance rules in natural language
        These are processed by the LLM for reasoning
        """
        return """
COMPLIANCE RULES FOR BUSINESS DOCUMENTS

=== INVOICE COMPLIANCE RULES ===

Rule 1: Valid Invoice Number
- Every invoice must have a unique invoice number/ID
- Format should be alphanumeric (e.g., INV-001, INV-2024-001)
- Must not be blank or "N/A"

Rule 2: Date Requirements
- Invoice date must be present and valid
- Invoice date should not be in the future
- Due date (if present) must be after invoice date

Rule 3: Financial Information
- Total amount must be present and greater than zero
- If tax is shown, verify: subtotal + tax = total (within rounding tolerance)
- Currency should be clearly indicated

Rule 4: Party Information
- Vendor/supplier name must be present
- Customer/buyer information should be present
- Contact information recommended

Rule 5: Payment Terms
- Payment terms should be clearly stated (e.g., "Net 30", "Due on receipt")
- Due date should align with payment terms

Rule 6: Line Items
- If line items present, each should have description and price
- Quantities should be positive numbers
- Line item totals should sum to subtotal

=== PURCHASE ORDER COMPLIANCE RULES ===

Rule 7: PO Number
- Must have unique PO number
- Should follow company format standards

Rule 8: Delivery Information
- Delivery date should be specified
- Delivery date should be reasonable (not in past, not too far future)
- Shipping address should be complete

Rule 9: Authorization
- PO should indicate authorized approver (if field exists)
- Amount should be within approval limits

=== GRN COMPLIANCE RULES ===

Rule 10: Receipt Documentation
- GRN must have unique receipt number
- Receipt date must be present
- Received by person should be identified

Rule 11: PO Matching
- GRN should reference corresponding PO number
- Quantities received should match PO quantities (or have variance explanation)

Rule 12: Quality Check
- Quality inspection status should be documented
- Any damages or discrepancies should be noted

=== GENERAL RULES ===

Rule 13: Data Completeness
- Critical fields should not be missing or empty
- Document should be machine-readable

Rule 14: Consistency
- Dates should be logically consistent
- Amounts should be mathematically correct
- References between documents should be valid

Rule 15: Compliance Threshold
- Documents with 2+ FAIL status are non-compliant
- Documents with 3+ WARNING status require review
- All CRITICAL rules must PASS

END OF RULES
"""
    
    def run_compliance_check(self, 
                            document: Dict[str, Any],
                            verbose: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive compliance check on a single document
        Returns structured compliance report
        """
        
        doc_id = document.get("document_id", document.get("source_file", "Unknown"))
        doc_type = document.get("document_type", "unknown")
        
        if verbose:
            print(f"\n⚖️  Running compliance check: {doc_id}")
            print(f"   Type: {doc_type}")
        
        # Use LLM to analyze compliance
        analysis = self.llm.compliance_analysis(document, self.compliance_rules)
        
        # Parse LLM response to extract structured data
        compliance_result = self._parse_compliance_response(analysis, document)
        
        if verbose:
            self._print_compliance_summary(compliance_result)
        
        return compliance_result
    
    def _parse_compliance_response(self, 
                                   llm_response: str,
                                   document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse LLM compliance analysis into structured format
        """
        
        # Count PASS/FAIL/WARNING in response
        pass_count = llm_response.upper().count("PASS")
        fail_count = llm_response.upper().count("FAIL")
        warning_count = llm_response.upper().count("WARNING")
        
        # Determine overall status
        if fail_count >= 2:
            overall_status = "NON-COMPLIANT"
        elif fail_count >= 1 or warning_count >= 3:
            overall_status = "REQUIRES REVIEW"
        else:
            overall_status = "COMPLIANT"
        
        return {
            "document_id": document.get("document_id", document.get("source_file")),
            "document_type": document.get("document_type"),
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "warning_count": warning_count,
            "detailed_analysis": llm_response,
            "document_data": document
        }
    
    def _print_compliance_summary(self, result: Dict[str, Any]):
        """
        Print formatted compliance summary to console
        """
        status = result["overall_status"]
        
        # Color coding
        if status == "COMPLIANT":
            status_icon = "✅"
        elif status == "REQUIRES REVIEW":
            status_icon = "⚠️"
        else:
            status_icon = "❌"
        
        print(f"\n{status_icon} Overall Status: {status}")
        print(f"   Pass: {result['pass_count']} | Fail: {result['fail_count']} | Warning: {result['warning_count']}")
    
    def run_batch_compliance(self, 
                            documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run compliance checks on multiple documents
        """
        print(f"\n⚖️  Running batch compliance check on {len(documents)} document(s)\n")
        
        results = []
        for i, doc in enumerate(documents, 1):
            print(f"[{i}/{len(documents)}]", end=" ")
            result = self.run_compliance_check(doc, verbose=True)
            results.append(result)
        
        # Generate summary
        self._print_batch_summary(results)
        
        return results
    
    def _print_batch_summary(self, results: List[Dict[str, Any]]):
        """
        Print summary of batch compliance check
        """
        compliant = sum(1 for r in results if r["overall_status"] == "COMPLIANT")
        review = sum(1 for r in results if r["overall_status"] == "REQUIRES REVIEW")
        non_compliant = sum(1 for r in results if r["overall_status"] == "NON-COMPLIANT")
        
        print("\n" + "="*50)
        print("BATCH COMPLIANCE SUMMARY")
        print("="*50)
        print(f"✅ Compliant: {compliant}")
        print(f"⚠️  Requires Review: {review}")
        print(f"❌ Non-Compliant: {non_compliant}")
        print(f"Total Documents: {len(results)}")
        print("="*50 + "\n")
    
    def save_compliance_report(self, 
                              result: Dict[str, Any],
                              output_dir: Optional[Path] = None) -> Path:
        """
        Save compliance report to file
        """
        if output_dir is None:
            output_dir = Path("compliance_reports")
        
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename
        doc_id = result["document_id"].replace("/", "-").replace("\\", "-")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"compliance_{doc_id}_{timestamp}.json"
        
        report_path = output_dir / filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Report saved: {report_path}")
        return report_path
    
    def generate_executive_summary(self, results: List[Dict[str, Any]]) -> str:
        """
        Generate executive summary using LLM
        """
        print("\n📊 Generating executive summary...")
        
        # Extract detailed analyses
        analyses = [r["detailed_analysis"] for r in results]
        
        summary = self.llm.generate_compliance_summary(analyses)
        
        return summary
    
    def auto_compliance_mode(self) -> List[Dict[str, Any]]:
        """
        Automatic compliance mode - the agentic behavior
        
        This method demonstrates the "GenAI agent" mindset:
        1. Check if documents need parsing
        2. Parse if needed
        3. Load into vector store
        4. Run compliance checks
        5. Generate reports
        """
        
        print("\n🤖 AUTO-COMPLIANCE MODE ACTIVATED\n")
        
        # Step 1: Check for documents
        print("📁 Checking for documents...")
        parsed_docs = self.parser.get_all_parsed_documents()
        
        if not parsed_docs:
            print("   No parsed documents found. Parsing now...")
            parsed_docs = self.parser.parse_all_documents()
        
        if not parsed_docs:
            print("❌ No documents to analyze. Please add PDFs to documents/ folder.")
            return []
        
        print(f"✅ Found {len(parsed_docs)} document(s)")
        
        # Step 2: Load into vector store (for Q&A capability)
        print("\n🔄 Loading documents into vector store...")
        self.vector_store.add_documents(parsed_docs)
        
        # Step 3: Run compliance checks
        results = self.run_batch_compliance(parsed_docs)
        
        # Step 4: Generate executive summary
        if len(results) > 1:
            summary = self.generate_executive_summary(results)
            print("\n📊 EXECUTIVE SUMMARY:")
            print("="*50)
            print(summary)
            print("="*50)
        
        # Step 5: Save reports
        print("\n💾 Saving compliance reports...")
        for result in results:
            self.save_compliance_report(result)
        
        return results


if __name__ == "__main__":
    # Test compliance engine
    engine = ComplianceEngine()
    
    # Test with sample document
    test_doc = {
        "document_id": "INV-TEST-001",
        "document_type": "invoice",
        "date": "01/15/2024",
        "vendor": "Test Vendor Inc.",
        "total": "1000.00"
    }
    
    result = engine.run_compliance_check(test_doc)
    print("\nTest completed")
