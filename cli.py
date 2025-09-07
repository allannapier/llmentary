"""
llmentary CLI - Command-line interface for LLM monitoring and drift detection

Provides commands for analyzing drift patterns, checking consistency,
and inspecting specific LLM interactions.
"""

import click
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
from collections import Counter, defaultdict

try:
    from .llmentary import SQLiteStorage, LLMCall, logger
except ImportError:
    # Handle case when running as script
    from llmentary import SQLiteStorage, LLMCall, logger

def get_storage(db_path: str = "llmentary.db") -> SQLiteStorage:
    """Get storage instance for the given database path"""
    storage = SQLiteStorage(db_path)
    if not Path(db_path).exists():
        click.echo(f"Database not found at {db_path}")
        click.echo("Make sure you have run some LLM calls with llmentary monitoring enabled.")
        raise click.Abort()
    return storage

def format_hash(hash_str: str, length: int = 8) -> str:
    """Format hash for display"""
    return hash_str[:length] if hash_str else 'unknown'

def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for display"""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

@click.group()
@click.version_option(version="0.2.0")
@click.option('--db-path', default="llmentary.db", help='Path to llmentary database')
@click.pass_context
def cli(ctx, db_path):
    """llmentary - Privacy-first LLM monitoring and drift detection"""
    ctx.ensure_object(dict)
    ctx.obj['db_path'] = db_path

@cli.command()
@click.option('--days', '-d', default=7, help='Number of days to analyze (default: 7)')
@click.option('--provider', '-p', help='Filter by provider (openai, anthropic, google)')
@click.option('--model', '-m', help='Filter by model name')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.option('--advanced', is_flag=True, help='Use advanced semantic drift detection')
@click.option('--detection-mode', type=click.Choice(['exact', 'semantic', 'hybrid']), default='hybrid', help='Drift detection mode')
@click.option('--threshold', type=float, help='Custom drift detection threshold (0.0-1.0)')
@click.pass_context
def report(ctx, days, provider, model, output_format, advanced, detection_mode, threshold):
    """Generate drift detection report with optional advanced analysis"""
    storage = get_storage(ctx.obj['db_path'])
    
    # Get all calls from the specified time period
    since = datetime.now() - timedelta(days=days)
    
    with sqlite3.connect(ctx.obj['db_path']) as conn:
        query = '''
            SELECT * FROM llm_calls 
            WHERE timestamp > ? 
        '''
        params = [since.isoformat()]
        
        if provider:
            query += ' AND provider = ?'
            params.append(provider)
            
        if model:
            query += ' AND model = ?'
            params.append(model)
            
        query += ' ORDER BY timestamp DESC'
        
        cursor = conn.cursor()
        cursor.execute(query, params)
        
        calls = []
        for row in cursor.fetchall():
            calls.append({
                'id': row[0],
                'input_hash': row[1],
                'output_hash': row[2],
                'input_text': row[3],
                'output_text': row[4],
                'model': row[5],
                'provider': row[6],
                'timestamp': datetime.fromisoformat(row[7]),
                'metadata': json.loads(row[8] or '{}')
            })
    
    if not calls:
        click.echo("No calls found in the specified time period.")
        return
    
    # Convert to LLMCall objects for advanced analysis
    llm_calls = []
    for call_data in calls:
        try:
            from llmentary import LLMCall
            llm_calls.append(LLMCall(
                input_hash=call_data['input_hash'],
                output_hash=call_data['output_hash'],
                input_text=call_data['input_text'],
                output_text=call_data['output_text'],
                model=call_data['model'],
                provider=call_data['provider'],
                timestamp=call_data['timestamp'],
                metadata=call_data['metadata']
            ))
        except Exception as e:
            click.echo(f"Warning: Failed to process call data: {e}")
    
    # Use advanced analysis if requested
    if advanced and llm_calls:
        try:
            import sys
            sys.path.append('.')
            from drift_detector import AdvancedDriftDetector, DriftDetectionConfig, DriftType
            
            # Configure detector
            mode_map = {
                'exact': DriftType.EXACT,
                'semantic': DriftType.SEMANTIC,
                'hybrid': DriftType.HYBRID
            }
            
            config = DriftDetectionConfig(detection_mode=mode_map[detection_mode])
            if threshold:
                if detection_mode == 'exact':
                    config.exact_threshold = threshold
                elif detection_mode == 'semantic':
                    config.semantic_threshold = threshold
                else:
                    config.hybrid_threshold = threshold
            
            detector = AdvancedDriftDetector(config)
            
            click.echo(f"🔬 Running advanced drift analysis (mode: {detection_mode})...")
            advanced_analysis = detector.analyze_drift_patterns(llm_calls, time_window_hours=days*24)
            
            if output_format == 'json':
                click.echo(json.dumps(advanced_analysis, indent=2, default=str))
            else:
                display_advanced_drift_report(advanced_analysis, days, detection_mode)
            return
            
        except ImportError as e:
            if 'sentence_transformers' in str(e).lower() or 'torch' in str(e).lower():
                click.echo("⚠️  Advanced drift detection requires sentence-transformers:")
                click.echo("    pip install sentence-transformers")
                click.echo("    Falling back to basic analysis...")
            else:
                click.echo(f"⚠️  Import error: {e}")
                click.echo("    Falling back to basic analysis...")
        except Exception as e:
            click.echo(f"⚠️  Advanced analysis failed: {e}")
            click.echo("    Falling back to basic analysis...")
    
    # Fallback to basic drift analysis
    drift_analysis = analyze_drift_patterns(calls)
    
    if output_format == 'json':
        click.echo(json.dumps(drift_analysis, indent=2, default=str))
    else:
        display_drift_report(drift_analysis, days)

def analyze_drift_patterns(calls: List[Dict]) -> Dict[str, Any]:
    """Analyze drift patterns in the call data"""
    # Group calls by input hash
    input_groups = defaultdict(list)
    for call in calls:
        input_groups[call['input_hash']].append(call)
    
    # Find drift cases (same input, different outputs)
    drift_cases = []
    consistency_by_model = defaultdict(lambda: {'total': 0, 'consistent': 0})
    provider_stats = defaultdict(lambda: {'total': 0, 'drift_cases': 0})
    
    for input_hash, group in input_groups.items():
        if len(group) > 1:
            unique_outputs = set(call['output_hash'] for call in group)
            
            # Track provider stats
            for call in group:
                provider_stats[call['provider']]['total'] += 1
                consistency_by_model[call['model']]['total'] += 1
            
            if len(unique_outputs) > 1:
                # This is a drift case
                drift_cases.append({
                    'input_hash': input_hash,
                    'calls': len(group),
                    'unique_outputs': len(unique_outputs),
                    'models': list(set(call['model'] for call in group)),
                    'providers': list(set(call['provider'] for call in group)),
                    'first_seen': min(call['timestamp'] for call in group),
                    'last_seen': max(call['timestamp'] for call in group),
                    'sample_calls': group[:3]  # Sample for display
                })
                
                for call in group:
                    provider_stats[call['provider']]['drift_cases'] += 1
            else:
                # Consistent responses
                for call in group:
                    consistency_by_model[call['model']]['consistent'] += 1
    
    # Calculate consistency scores
    for model_stats in consistency_by_model.values():
        model_stats['consistency_score'] = (
            model_stats['consistent'] / model_stats['total'] 
            if model_stats['total'] > 0 else 0
        )
    
    return {
        'summary': {
            'total_calls': len(calls),
            'unique_inputs': len(input_groups),
            'drift_cases': len(drift_cases),
            'drift_rate': len(drift_cases) / len(input_groups) if input_groups else 0
        },
        'drift_cases': sorted(drift_cases, key=lambda x: x['last_seen'], reverse=True),
        'consistency_by_model': dict(consistency_by_model),
        'provider_stats': dict(provider_stats)
    }

def display_advanced_drift_report(analysis: Dict[str, Any], days: int, detection_mode: str):
    """Display formatted advanced drift report"""
    summary = analysis['summary']
    
    click.echo(f"\n🔬 llmentary Advanced Drift Report - Last {days} days")
    click.echo("=" * 60)
    click.echo(f"Detection Mode: {detection_mode.upper()}")
    
    # Summary section
    click.echo(f"\n📊 Summary:")
    click.echo(f"  Total comparisons: {summary['total_comparisons']:,}")
    click.echo(f"  Drift cases detected: {summary['drift_cases']:,}")
    click.echo(f"  Drift rate: {summary['drift_rate']:.1%}")
    click.echo(f"  Average similarity: {summary['average_similarity']:.3f}")
    
    # Severity breakdown
    if analysis['severity_breakdown']:
        click.echo(f"\n🚨 Severity Breakdown:")
        click.echo("-" * 25)
        
        severity_emojis = {
            'low': '🟢',
            'medium': '🟡', 
            'high': '🟠',
            'critical': '🔴'
        }
        
        for severity, count in analysis['severity_breakdown'].items():
            if count > 0:
                emoji = severity_emojis.get(severity, '⚪')
                click.echo(f"  {emoji} {severity.title()}: {count:,} cases")
    
    # Model drift rates
    if analysis['model_drift_rates']:
        click.echo(f"\n🤖 Model Drift Analysis:")
        click.echo("-" * 30)
        
        for model, stats in analysis['model_drift_rates'].items():
            click.echo(f"  {model}:")
            click.echo(f"    Total: {stats['total']:,} comparisons")
            click.echo(f"    Drift: {stats['drift']:,} cases ({stats['rate']:.1%})")
    
    # Provider drift rates
    if analysis['provider_drift_rates']:
        click.echo(f"\n🏢 Provider Drift Analysis:")
        click.echo("-" * 30)
        
        for provider, stats in analysis['provider_drift_rates'].items():
            click.echo(f"  {provider}:")
            click.echo(f"    Total: {stats['total']:,} comparisons")
            click.echo(f"    Drift: {stats['drift']:,} cases ({stats['rate']:.1%})")
    
    # Top drift cases
    if analysis['drift_results']:
        click.echo(f"\n⚠️  Top Drift Cases:")
        click.echo("-" * 25)
        
        # Sort by severity and similarity score
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        sorted_cases = sorted(
            analysis['drift_results'][:10],
            key=lambda x: (severity_order.get(x['severity'], 4), x['similarity_score'])
        )
        
        for i, case in enumerate(sorted_cases, 1):
            severity_emoji = {'low': '🟢', 'medium': '🟡', 'high': '🟠', 'critical': '🔴'}.get(case['severity'], '⚪')
            click.echo(f"\n{i}. {severity_emoji} Input Hash: {case['input_hash']}")
            click.echo(f"   Similarity: {case['similarity_score']:.3f} | Confidence: {case['confidence']:.3f}")
            click.echo(f"   Model: {case['model']} | Provider: {case['provider']}")

def display_drift_report(analysis: Dict[str, Any], days: int):
    """Display formatted drift report"""
    summary = analysis['summary']
    
    click.echo(f"\n🔍 llmentary Drift Report - Last {days} days")
    click.echo("=" * 50)
    
    # Summary section
    click.echo(f"\n📊 Summary:")
    click.echo(f"  Total LLM calls: {summary['total_calls']:,}")
    click.echo(f"  Unique inputs: {summary['unique_inputs']:,}")
    click.echo(f"  Drift cases detected: {summary['drift_cases']:,}")
    click.echo(f"  Drift rate: {summary['drift_rate']:.1%}")
    
    # Drift cases
    if analysis['drift_cases']:
        click.echo(f"\n⚠️  Top Drift Cases:")
        click.echo("-" * 30)
        
        for i, case in enumerate(analysis['drift_cases'][:5]):
            click.echo(f"\n{i+1}. Input Hash: {format_hash(case['input_hash'])}")
            click.echo(f"   Calls: {case['calls']} | Unique outputs: {case['unique_outputs']}")
            click.echo(f"   Models: {', '.join(case['models'])}")
            click.echo(f"   First seen: {format_timestamp(case['first_seen'])}")
            click.echo(f"   Last seen: {format_timestamp(case['last_seen'])}")
    
    # Model consistency
    if analysis['consistency_by_model']:
        click.echo(f"\n📈 Model Consistency:")
        click.echo("-" * 25)
        
        for model, stats in analysis['consistency_by_model'].items():
            score = stats['consistency_score']
            click.echo(f"  {model}: {score:.1%} ({stats['consistent']}/{stats['total']})")
    
    # Provider stats
    if analysis['provider_stats']:
        click.echo(f"\n🏢 Provider Statistics:")
        click.echo("-" * 25)
        
        for provider, stats in analysis['provider_stats'].items():
            drift_rate = stats['drift_cases'] / stats['total'] if stats['total'] > 0 else 0
            click.echo(f"  {provider}: {stats['total']} calls, {drift_rate:.1%} drift rate")

@cli.command()
@click.option('--provider', '-p', help='Filter by provider')
@click.option('--model', '-m', help='Filter by model name') 
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.pass_context
def consistency(ctx, provider, model, output_format):
    """Check consistency scores across models and providers"""
    storage = get_storage(ctx.obj['db_path'])
    
    with sqlite3.connect(ctx.obj['db_path']) as conn:
        query = 'SELECT * FROM llm_calls'
        params = []
        
        if provider or model:
            query += ' WHERE '
            conditions = []
            if provider:
                conditions.append('provider = ?')
                params.append(provider)
            if model:
                conditions.append('model = ?')
                params.append(model)
            query += ' AND '.join(conditions)
        
        cursor = conn.cursor()
        cursor.execute(query, params)
        
        calls = []
        for row in cursor.fetchall():
            calls.append({
                'input_hash': row[1],
                'output_hash': row[2],
                'model': row[5],
                'provider': row[6],
                'timestamp': datetime.fromisoformat(row[7])
            })
    
    if not calls:
        click.echo("No calls found matching the criteria.")
        return
    
    # Calculate consistency scores
    scores = calculate_consistency_scores(calls)
    
    if output_format == 'json':
        click.echo(json.dumps(scores, indent=2))
    else:
        display_consistency_scores(scores)

def calculate_consistency_scores(calls: List[Dict]) -> Dict[str, Any]:
    """Calculate consistency scores for models and providers"""
    # Group by input hash
    input_groups = defaultdict(list)
    for call in calls:
        input_groups[call['input_hash']].append(call)
    
    model_consistency = defaultdict(lambda: {'total_comparisons': 0, 'consistent_responses': 0})
    provider_consistency = defaultdict(lambda: {'total_comparisons': 0, 'consistent_responses': 0})
    
    for input_hash, group in input_groups.items():
        if len(group) > 1:
            # Group by model/provider
            by_model = defaultdict(list)
            by_provider = defaultdict(list)
            
            for call in group:
                by_model[call['model']].append(call)
                by_provider[call['provider']].append(call)
            
            # Check consistency within each model
            for model, model_calls in by_model.items():
                if len(model_calls) > 1:
                    unique_outputs = set(call['output_hash'] for call in model_calls)
                    model_consistency[model]['total_comparisons'] += 1
                    if len(unique_outputs) == 1:
                        model_consistency[model]['consistent_responses'] += 1
            
            # Check consistency within each provider
            for provider, provider_calls in by_provider.items():
                if len(provider_calls) > 1:
                    unique_outputs = set(call['output_hash'] for call in provider_calls)
                    provider_consistency[provider]['total_comparisons'] += 1
                    if len(unique_outputs) == 1:
                        provider_consistency[provider]['consistent_responses'] += 1
    
    # Calculate final scores
    model_scores = {}
    for model, stats in model_consistency.items():
        model_scores[model] = {
            'consistency_score': stats['consistent_responses'] / stats['total_comparisons'] if stats['total_comparisons'] > 0 else 0,
            'total_comparisons': stats['total_comparisons'],
            'consistent_responses': stats['consistent_responses']
        }
    
    provider_scores = {}
    for provider, stats in provider_consistency.items():
        provider_scores[provider] = {
            'consistency_score': stats['consistent_responses'] / stats['total_comparisons'] if stats['total_comparisons'] > 0 else 0,
            'total_comparisons': stats['total_comparisons'],
            'consistent_responses': stats['consistent_responses']
        }
    
    return {
        'models': model_scores,
        'providers': provider_scores,
        'overall': {
            'total_calls': len(calls),
            'unique_inputs': len(input_groups)
        }
    }

def display_consistency_scores(scores: Dict[str, Any]):
    """Display formatted consistency scores"""
    click.echo("\n📊 llmentary Consistency Report")
    click.echo("=" * 40)
    
    overall = scores['overall']
    click.echo(f"\n🎯 Overall Statistics:")
    click.echo(f"  Total calls analyzed: {overall['total_calls']:,}")
    click.echo(f"  Unique inputs: {overall['unique_inputs']:,}")
    
    # Model consistency
    if scores['models']:
        click.echo(f"\n🤖 Model Consistency Scores:")
        click.echo("-" * 30)
        
        sorted_models = sorted(scores['models'].items(), 
                              key=lambda x: x[1]['consistency_score'], 
                              reverse=True)
        
        for model, stats in sorted_models:
            score = stats['consistency_score']
            click.echo(f"  {model:<25} {score:>6.1%} ({stats['consistent_responses']}/{stats['total_comparisons']})")
    
    # Provider consistency  
    if scores['providers']:
        click.echo(f"\n🏢 Provider Consistency Scores:")
        click.echo("-" * 30)
        
        sorted_providers = sorted(scores['providers'].items(),
                                 key=lambda x: x[1]['consistency_score'],
                                 reverse=True)
        
        for provider, stats in sorted_providers:
            score = stats['consistency_score']
            click.echo(f"  {provider:<25} {score:>6.1%} ({stats['consistent_responses']}/{stats['total_comparisons']})")

@cli.command()
@click.argument('input_hash')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.option('--show-text', is_flag=True, help='Show actual input/output text (if stored)')
@click.pass_context
def inspect(ctx, input_hash, output_format, show_text):
    """Deep dive into specific input hash"""
    storage = get_storage(ctx.obj['db_path'])
    
    with sqlite3.connect(ctx.obj['db_path']) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM llm_calls WHERE input_hash LIKE ? ORDER BY timestamp
        ''', (f"{input_hash}%",))
        
        calls = []
        for row in cursor.fetchall():
            calls.append({
                'id': row[0],
                'input_hash': row[1],
                'output_hash': row[2],
                'input_text': row[3],
                'output_text': row[4],
                'model': row[5],
                'provider': row[6],
                'timestamp': datetime.fromisoformat(row[7]),
                'metadata': json.loads(row[8] or '{}')
            })
    
    if not calls:
        click.echo(f"No calls found for input hash starting with: {input_hash}")
        return
    
    if output_format == 'json':
        call_data = [{
            'id': call['id'],
            'input_hash': call['input_hash'],
            'output_hash': call['output_hash'],
            'model': call['model'],
            'provider': call['provider'],
            'timestamp': call['timestamp'].isoformat(),
            'metadata': call['metadata'],
            'input_text': call['input_text'] if show_text else None,
            'output_text': call['output_text'] if show_text else None
        } for call in calls]
        click.echo(json.dumps(call_data, indent=2))
    else:
        display_inspection_results(calls, show_text)

def display_inspection_results(calls: List[Dict], show_text: bool):
    """Display formatted inspection results"""
    if not calls:
        return
    
    first_call = calls[0]
    click.echo(f"\n🔍 llmentary Call Inspection")
    click.echo("=" * 40)
    
    click.echo(f"\nInput Hash: {first_call['input_hash']}")
    click.echo(f"Total calls: {len(calls)}")
    click.echo(f"Unique outputs: {len(set(call['output_hash'] for call in calls))}")
    
    # Show input text if available and requested
    if show_text and first_call['input_text']:
        click.echo(f"\n📝 Input Text:")
        click.echo(f"  {first_call['input_text']}")
    
    click.echo(f"\n📋 Call History:")
    click.echo("-" * 20)
    
    for i, call in enumerate(calls, 1):
        click.echo(f"\n{i}. {format_timestamp(call['timestamp'])}")
        click.echo(f"   Provider: {call['provider']} | Model: {call['model']}")
        click.echo(f"   Output Hash: {format_hash(call['output_hash'], 16)}")
        
        if show_text and call['output_text']:
            click.echo(f"   Output: {call['output_text'][:100]}{'...' if len(call['output_text']) > 100 else ''}")
        
        if call['metadata']:
            click.echo(f"   Metadata: {json.dumps(call['metadata'], default=str)}")

@cli.command()
@click.option('--store-raw-text', type=bool, help='Enable/disable storing raw text (true/false)')
@click.option('--drift-threshold', type=float, help='Set drift detection threshold (0.0-1.0)')
@click.option('--advanced-drift', type=bool, help='Enable/disable advanced drift detection')
@click.option('--detection-mode', type=click.Choice(['exact', 'semantic', 'hybrid']), help='Set drift detection mode')
@click.option('--semantic-threshold', type=float, help='Set semantic similarity threshold (0.0-1.0)')
@click.option('--list', 'list_config', is_flag=True, help='List current configuration')
@click.pass_context
def config(ctx, store_raw_text, drift_threshold, advanced_drift, detection_mode, semantic_threshold, list_config):
    """Manage llmentary configuration settings"""
    
    if list_config:
        # Show current configuration
        try:
            from llmentary import monitor
            config = monitor.config
            click.echo("\n⚙️  llmentary Configuration")
            click.echo("=" * 30)
            click.echo(f"Store raw text: {config.get('store_raw_text', False)}")
            click.echo(f"Drift threshold: {config.get('drift_threshold', 0.85)}")
            click.echo(f"Storage backend: {config.get('storage_backend', 'sqlite')}")
            click.echo(f"Database path: {config.get('db_path', 'llmentary.db')}")
            
            # Advanced drift detection settings
            click.echo(f"\n🔬 Advanced Drift Detection:")
            click.echo(f"  Enabled: {config.get('advanced_drift_detection', False)}")
            click.echo(f"  Detection mode: {config.get('drift_detection_mode', 'hybrid')}")
            click.echo(f"  Semantic threshold: {config.get('semantic_threshold', 0.85)}")
            click.echo(f"  Exact threshold: {config.get('exact_threshold', 1.0)}")
            click.echo(f"  Hybrid threshold: {config.get('hybrid_threshold', 0.80)}")
            
            # Check if database exists and show stats
            db_path = config.get('db_path', 'llmentary.db')
            if Path(db_path).exists():
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM llm_calls")
                    total_calls = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(DISTINCT provider) FROM llm_calls")
                    providers = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(DISTINCT model) FROM llm_calls")
                    models = cursor.fetchone()[0]
                    
                    click.echo(f"\n📊 Database Statistics:")
                    click.echo(f"Total calls: {total_calls:,}")
                    click.echo(f"Providers: {providers}")
                    click.echo(f"Models: {models}")
            else:
                click.echo(f"\n⚠️  Database not found: {db_path}")
                
        except Exception as e:
            click.echo(f"Error reading configuration: {e}")
        return
    
    # Update configuration
    updates = {}
    if store_raw_text is not None:
        updates['store_raw_text'] = store_raw_text
    if drift_threshold is not None:
        if not 0.0 <= drift_threshold <= 1.0:
            click.echo("Error: Drift threshold must be between 0.0 and 1.0")
            return
        updates['drift_threshold'] = drift_threshold
    if advanced_drift is not None:
        updates['advanced_drift_detection'] = advanced_drift
    if detection_mode is not None:
        updates['drift_detection_mode'] = detection_mode
    if semantic_threshold is not None:
        if not 0.0 <= semantic_threshold <= 1.0:
            click.echo("Error: Semantic threshold must be between 0.0 and 1.0")
            return
        updates['semantic_threshold'] = semantic_threshold
    
    if not updates:
        click.echo("No configuration changes specified. Use --list to see current config.")
        return
    
    try:
        from llmentary import monitor
        monitor.configure(**updates)
        
        click.echo("\n✅ Configuration updated:")
        for key, value in updates.items():
            click.echo(f"  {key}: {value}")
            
    except Exception as e:
        click.echo(f"Error updating configuration: {e}")

@cli.command()
@click.pass_context 
def status(ctx):
    """Show llmentary system status and statistics"""
    db_path = ctx.obj['db_path']
    
    click.echo("\n📊 llmentary System Status")
    click.echo("=" * 35)
    
    # Check database
    if not Path(db_path).exists():
        click.echo(f"❌ Database not found: {db_path}")
        click.echo("Run some LLM calls with llmentary monitoring to create the database.")
        return
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Basic stats
            cursor.execute("SELECT COUNT(*) FROM llm_calls")
            total_calls = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT input_hash) FROM llm_calls") 
            unique_inputs = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT provider) FROM llm_calls")
            providers = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT model) FROM llm_calls")
            models = cursor.fetchone()[0]
            
            # Recent activity
            cursor.execute("""
                SELECT COUNT(*) FROM llm_calls 
                WHERE timestamp > datetime('now', '-1 day')
            """)
            calls_last_24h = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM llm_calls 
                WHERE timestamp > datetime('now', '-1 hour')
            """)
            calls_last_hour = cursor.fetchone()[0]
            
            click.echo(f"✅ Database: {db_path}")
            click.echo(f"\n📈 Overall Statistics:")
            click.echo(f"  Total LLM calls: {total_calls:,}")
            click.echo(f"  Unique inputs: {unique_inputs:,}")
            click.echo(f"  Providers monitored: {providers}")
            click.echo(f"  Models used: {models}")
            
            click.echo(f"\n⏰ Recent Activity:")
            click.echo(f"  Last 24 hours: {calls_last_24h:,} calls")
            click.echo(f"  Last hour: {calls_last_hour:,} calls")
            
            # Provider breakdown
            cursor.execute("""
                SELECT provider, COUNT(*) as call_count 
                FROM llm_calls 
                GROUP BY provider 
                ORDER BY call_count DESC
            """)
            provider_stats = cursor.fetchall()
            
            if provider_stats:
                click.echo(f"\n🏢 Provider Breakdown:")
                for provider, count in provider_stats:
                    click.echo(f"  {provider}: {count:,} calls")
            
            # Check for recent drift
            cursor.execute("""
                SELECT input_hash, COUNT(DISTINCT output_hash) as unique_outputs
                FROM llm_calls 
                WHERE timestamp > datetime('now', '-7 days')
                GROUP BY input_hash 
                HAVING unique_outputs > 1
                ORDER BY unique_outputs DESC
                LIMIT 5
            """)
            recent_drift = cursor.fetchall()
            
            if recent_drift:
                click.echo(f"\n⚠️  Recent Drift Cases (Last 7 days):")
                for input_hash, unique_outputs in recent_drift:
                    click.echo(f"  {format_hash(input_hash)}: {unique_outputs} different outputs")
            else:
                click.echo(f"\n✅ No drift detected in the last 7 days")
                
    except Exception as e:
        click.echo(f"❌ Error accessing database: {e}")

@cli.group()
def train():
    """Training mode commands"""
    pass

@train.command('start')
@click.option('--category', '-c', help='Category for training examples')
@click.option('--non-interactive', is_flag=True, help='Save all interactions automatically')
@click.pass_context  
def start_training(ctx, category, non_interactive):
    """Start interactive training mode"""
    try:
        import sys
        sys.path.append('.')
        from training_tester import training_mode, get_trainer_tester
    except ImportError:
        click.echo("❌ Training mode not available. Please check installation.")
        return
    
    click.echo("🎓 llmentary Training Mode")
    click.echo("=" * 30)
    click.echo("Run your application now. When you make LLM calls,")
    click.echo("you'll be prompted to save Q&A pairs you approve of.")
    click.echo("Press Ctrl+C to exit training mode.")
    click.echo()
    
    trainer = get_trainer_tester(ctx.obj['db_path'].replace('.db', '_training.db'))
    trainer.training_mode = True
    trainer.interactive = not non_interactive
    
    click.echo("✅ Training mode active!")
    if non_interactive:
        click.echo("🤖 Auto-saving all interactions")
    else:
        click.echo("💬 Interactive mode - you'll be prompted for each Q&A")
    
    # Keep the process alive to maintain training mode
    try:
        while True:
            click.pause(info="Training mode active. Press any key to check status, Ctrl+C to exit.")
            stats = trainer.get_training_stats()
            click.echo(f"📊 Current training data: {stats.get('total', 0)} examples")
    except KeyboardInterrupt:
        trainer.training_mode = False
        click.echo("\n🛑 Training mode stopped")

@train.command('list')
@click.option('--category', '-c', help='Filter by category')
@click.option('--model', '-m', help='Filter by model')
@click.option('--provider', '-p', help='Filter by provider')
@click.option('--limit', '-l', default=10, help='Limit number of results')
@click.pass_context
def list_examples(ctx, category, model, provider, limit):
    """List saved training examples"""
    try:
        import sys
        sys.path.append('.')
        from training_tester import get_trainer_tester
    except ImportError:
        click.echo("❌ Training functionality not available.")
        return
    
    trainer = get_trainer_tester(ctx.obj['db_path'].replace('.db', '_training.db'))
    examples = trainer.storage.get_examples(category=category, model=model, provider=provider)
    
    if not examples:
        click.echo("📝 No training examples found.")
        if any([category, model, provider]):
            click.echo("Try adjusting your filters.")
        else:
            click.echo("Run 'llmentary train start' to begin collecting examples.")
        return
    
    click.echo(f"\n📚 Training Examples ({len(examples)} total)")
    click.echo("=" * 40)
    
    for i, example in enumerate(examples[:limit], 1):
        click.echo(f"\n{i}. ID: {example.id}")
        click.echo(f"   Question: {example.question[:60]}{'...' if len(example.question) > 60 else ''}")
        click.echo(f"   Model: {example.model} | Provider: {example.provider}")
        if example.category:
            click.echo(f"   Category: {example.category}")
        click.echo(f"   Created: {example.created_at.strftime('%Y-%m-%d %H:%M')}")
        
    if len(examples) > limit:
        click.echo(f"\n... and {len(examples) - limit} more examples")

@train.command('stats')
@click.pass_context
def training_stats(ctx):
    """Show training statistics"""
    try:
        import sys
        sys.path.append('.')
        from training_tester import get_trainer_tester
    except ImportError:
        click.echo("❌ Training functionality not available.")
        return
    
    trainer = get_trainer_tester(ctx.obj['db_path'].replace('.db', '_training.db'))
    stats = trainer.get_training_stats()
    
    if stats.get('total', 0) == 0:
        click.echo("📝 No training examples collected yet.")
        click.echo("Run 'llmentary train start' to begin collecting examples.")
        return
    
    click.echo(f"\n📊 llmentary Training Statistics")
    click.echo("=" * 40)
    click.echo(f"Total examples: {stats['total']:,}")
    
    if stats.get('categories'):
        click.echo(f"\n📂 Categories:")
        for category, count in stats['categories'].items():
            click.echo(f"  {category}: {count:,}")
    
    if stats.get('models'):
        click.echo(f"\n🤖 Models:")
        for model, count in stats['models'].items():
            click.echo(f"  {model}: {count:,}")
    
    if stats.get('providers'):
        click.echo(f"\n🏢 Providers:")
        for provider, count in stats['providers'].items():
            click.echo(f"  {provider}: {count:,}")
    
    if stats.get('oldest') and stats.get('newest'):
        click.echo(f"\n📅 Date Range:")
        click.echo(f"  Oldest: {stats['oldest'].strftime('%Y-%m-%d %H:%M')}")
        click.echo(f"  Newest: {stats['newest'].strftime('%Y-%m-%d %H:%M')}")

@cli.command()
@click.option('--category', '-c', help='Test only this category')
@click.option('--model', '-m', help='Test only this model')
@click.option('--provider', '-p', help='Test only this provider')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed results')
@click.pass_context
def test(ctx, category, model, provider, verbose):
    """Run regression tests against saved training examples"""
    try:
        import sys
        sys.path.append('.')
        from training_tester import get_trainer_tester
    except ImportError:
        click.echo("❌ Testing functionality not available.")
        return
    
    trainer = get_trainer_tester(ctx.obj['db_path'].replace('.db', '_training.db'))
    examples = trainer.storage.get_examples(category=category, model=model, provider=provider)
    
    if not examples:
        click.echo("📝 No training examples found for testing.")
        if any([category, model, provider]):
            click.echo("Try adjusting your filters or run without filters.")
        else:
            click.echo("Run 'llmentary train start' to collect training examples first.")
        return
    
    click.echo(f"\n🧪 Running regression tests ({len(examples)} examples)")
    click.echo("=" * 50)
    click.echo()
    click.echo("⚠️  NOTE: This is a framework for regression testing.")
    click.echo("You need to integrate llmentary into your application")
    click.echo("to actually run the tests against current LLM responses.")
    click.echo()
    click.echo("Example integration:")
    click.echo("```python")
    click.echo("from llmentary.training_tester import capture_interaction")
    click.echo()
    click.echo("# In your app:")
    click.echo("response = llm.ask(question)")
    click.echo("capture_interaction(question, response, model, provider)")
    click.echo("```")
    click.echo()
    
    # Show what would be tested
    click.echo("📋 Test Plan:")
    categories = {}
    for example in examples:
        cat = example.category or "uncategorized"
        categories[cat] = categories.get(cat, 0) + 1
    
    for category_name, count in categories.items():
        click.echo(f"  {category_name}: {count} examples")

@cli.command() 
@click.option('--question', '-q', required=True, help='Question to ask')
@click.option('--model', '-m', default='unknown', help='Model name')
@click.option('--provider', '-p', default='unknown', help='Provider name')
@click.option('--category', '-c', help='Category for this example')
@click.pass_context
def add_example(ctx, question, model, provider, category):
    """Manually add a training example"""
    try:
        import sys
        sys.path.append('.')
        from training_tester import get_trainer_tester
    except ImportError:
        click.echo("❌ Training functionality not available.")
        return
    
    click.echo("🤖 Please provide the expected answer for this question:")
    answer = click.prompt("Answer", type=str)
    
    trainer = get_trainer_tester(ctx.obj['db_path'].replace('.db', '_training.db'))
    
    # Force training mode for manual addition
    old_mode = trainer.training_mode
    trainer.training_mode = True
    
    try:
        example_id = trainer.capture_interaction(
            question=question,
            answer=answer,
            model=model,
            provider=provider,
            category=category,
            force_save=True
        )
    finally:
        trainer.training_mode = old_mode
    
    if example_id:
        click.echo(f"✅ Training example {example_id} saved successfully!")
    else:
        click.echo("❌ Failed to save training example.")

if __name__ == '__main__':
    cli()