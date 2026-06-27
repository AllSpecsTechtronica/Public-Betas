# Session Tracker - For GPT-4o Billing and Usage
# ═══════════════════════════════════════════════════════════════════════════════

import json
import os
import time
from datetime import datetime
from utils.debug import debug_print

BASE_CV_PROGRAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEBUG_DIR = os.path.join(BASE_CV_PROGRAM_DIR, "debug")

class SessionTracker:
    """Track session duration, API calls, and costs with GPT-4o pricing"""
    
    def __init__(self):
        self.session_start = datetime.now()
        self.analyses_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.session_id = int(time.time())
        
        # GPT-4o Pricing (in cents per token for precision)
        self.pricing = {
            'gpt-4o': {
                'input': 0.00025,  # $2.50 per 1M tokens = 0.00025 cents per token
                'output': 0.00025  # Same for regular gpt-4o
            },
            'gpt-4o-2024-08-06': {
                'input': 0.00025,   # $2.50 per 1M tokens = 0.00025 cents per token
                'output': 0.001     # $10.00 per 1M tokens = 0.001 cents per token
            }
        }
        
        self.analyses_history = []
        debug_print(f"[SESSION_TRACKER] Session {self.session_id} started at {self.session_start.strftime('%H:%M:%S')}")
    
    def get_session_duration(self):
        """Get session duration in minutes"""
        return (datetime.now() - self.session_start).total_seconds() / 60
    
    def calculate_cost(self, model, input_tokens, output_tokens):
        """Calculate cost in cents for given token usage"""
        if model not in self.pricing:
            model = 'gpt-4o'  # Default fallback
        
        input_cost = input_tokens * self.pricing[model]['input']
        output_cost = output_tokens * self.pricing[model]['output']
        total_cost = input_cost + output_cost
        
        return {
            'input_cost_cents': input_cost,
            'output_cost_cents': output_cost, 
            'total_cost_cents': total_cost,
            'total_cost_dollars': total_cost / 100
        }
    
    def record_analysis(self, model, input_tokens, output_tokens, analysis_text, rect_id):
        """Record a completed analysis with token usage and cost"""
        self.analyses_count += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        cost_info = self.calculate_cost(model, input_tokens, output_tokens)
        
        analysis_record = {
            'analysis_id': self.analyses_count,
            'rect_id': rect_id,
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'cost_info': cost_info,
            'analysis_preview': analysis_text[:100] + '...' if len(analysis_text) > 100 else analysis_text
        }
        
        self.analyses_history.append(analysis_record)
        
        # Print real-time cost info
        duration_min = self.get_session_duration()
        total_session_cost = self.get_total_session_cost()
        
        debug_print(f"[SESSION_TRACKER] Analysis #{self.analyses_count} complete - {input_tokens + output_tokens} tokens | Cost: {cost_info['total_cost_cents']:.4f} cents")
        debug_print(f"[SESSION_TRACKER] Session: {duration_min:.1f}min | {self.analyses_count} analyses | {self.total_input_tokens + self.total_output_tokens:,} tokens | ${total_session_cost['total_cost_dollars']:.4f}")
        
        return analysis_record
    
    def get_total_session_cost(self):
        """Calculate total session cost across all analyses"""
        total_input_cost = 0
        total_output_cost = 0
        
        for analysis in self.analyses_history:
            cost_info = analysis['cost_info']
            total_input_cost += cost_info['input_cost_cents']
            total_output_cost += cost_info['output_cost_cents']
        
        total_cost_cents = total_input_cost + total_output_cost
        
        return {
            'input_cost_cents': total_input_cost,
            'output_cost_cents': total_output_cost,
            'total_cost_cents': total_cost_cents,
            'total_cost_dollars': total_cost_cents / 100
        }
    
    def save_session_report(self):
        """Save comprehensive session report to debug folder"""
        duration_min = self.get_session_duration()
        total_cost = self.get_total_session_cost()
        
        report_data = {
            'session_info': {
                'session_id': self.session_id,
                'start_time': self.session_start.isoformat(),
                'duration_minutes': duration_min,
                'analyses_count': self.analyses_count,
                'total_input_tokens': self.total_input_tokens,
                'total_output_tokens': self.total_output_tokens,
                'total_tokens': self.total_input_tokens + self.total_output_tokens
            },
            'cost_summary': total_cost,
            'analyses_history': self.analyses_history,
            'pricing_used': self.pricing
        }
        
        # Create debug directory if it doesn't exist
        os.makedirs(DEBUG_DIR, exist_ok=True)
        
        # Save JSON report
        json_path = os.path.join(DEBUG_DIR, f"session_report_{self.session_id}.json")
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            debug_print(f"[SESSION_TRACKER] Report saved to: {json_path}")
        except Exception as e:
            debug_print(f"[SESSION_TRACKER] Error saving report: {e}")
        
        # Save human-readable report  
        txt_path = os.path.join(DEBUG_DIR, f"session_summary_{self.session_id}.txt")
        summary = f"""SESSION SUMMARY REPORT
========================================
Session ID: {self.session_id}
Start Time: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration_min:.1f} minutes
Total Analyses: {self.analyses_count}

TOKEN USAGE:
- Input Tokens: {self.total_input_tokens:,}
- Output Tokens: {self.total_output_tokens:,}  
- Total Tokens: {self.total_input_tokens + self.total_output_tokens:,}

COST BREAKDOWN:
- Input Cost: {total_cost['input_cost_cents']:.4f} cents
- Output Cost: {total_cost['output_cost_cents']:.4f} cents
- Total Cost: {total_cost['total_cost_cents']:.4f} cents (${total_cost['total_cost_dollars']:.4f})

PERFORMANCE METRICS:
- Average tokens per analysis: {(self.total_input_tokens + self.total_output_tokens) / max(1, self.analyses_count):.1f}
- Cost per analysis: {total_cost['total_cost_cents'] / max(1, self.analyses_count):.4f} cents
- Analysis rate: {self.analyses_count / max(0.1, duration_min):.1f} analyses/minute
========================================"""
        
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            debug_print(f"[SESSION_TRACKER] Summary saved to: {txt_path}")
        except Exception as e:
            debug_print(f"[SESSION_TRACKER] Error saving summary: {e}")
        
        return report_data
