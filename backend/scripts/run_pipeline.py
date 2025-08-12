"""
CLI Script for Timbre Data Ingestion Pipeline
Provides command-line interface using Typer
"""
import asyncio
import sys
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rprint

# Add the parent directory to the path so we can import from app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipelines import run_pipeline_sync
from app.config import settings

app = typer.Typer(
    name="timbre-pipeline",
    help="Timbre Data Ingestion Pipeline CLI",
    add_completion=False
)
console = Console()


@app.command()
def run(
    max_songs: Optional[int] = typer.Option(
        None,
        "--max-songs",
        "-n",
        help=f"Maximum number of songs to process (default: {settings.max_songs})"
    ),
    skip_aoty: bool = typer.Option(
        False,
        "--skip-aoty",
        help="Skip AOTY scraping for faster execution"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Process data but don't write to database"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    )
):
    """
    Run the complete ingestion pipeline
    
    This will:
    1. Pull top tracks from Last.fm
    2. Deduplicate tracks using canonical IDs
    3. Enrich with Spotify metadata
    4. Scrape AOTY ratings (unless --skip-aoty)
    5. Insert all data into the database (unless --dry-run)
    """
    
    if max_songs is None:
        max_songs = settings.max_songs
    
    # Display configuration
    config_table = Table(title="Pipeline Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    
    config_table.add_row("Max Songs", str(max_songs))
    config_table.add_row("Skip AOTY", "Yes" if skip_aoty else "No")
    config_table.add_row("Dry Run", "Yes" if dry_run else "No")
    config_table.add_row("Verbose", "Yes" if verbose else "No")
    
    console.print(config_table)
    console.print()
    
    # Confirm execution unless dry run
    if not dry_run:
        confirm = typer.confirm(
            f"This will process {max_songs} songs and write to the database. Continue?"
        )
        if not confirm:
            rprint("[yellow]Pipeline cancelled by user[/yellow]")
            raise typer.Exit(1)
    
    # Run pipeline with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Running ingestion pipeline...", total=None)
        
        try:
            result = run_pipeline_sync(
                max_songs=max_songs,
                skip_aoty=skip_aoty,
                dry_run=dry_run
            )
            
            progress.remove_task(task)
            
            # Display results
            _display_results(result)
            
            # Exit with appropriate code
            if result.success:
                rprint("[green]✓ Pipeline completed successfully![/green]")
                raise typer.Exit(0)
            else:
                rprint("[red]✗ Pipeline failed![/red]")
                raise typer.Exit(1)
                
        except KeyboardInterrupt:
            try:
                progress.remove_task(task)
            except KeyError:
                pass  # Task already removed or doesn't exist
            rprint("[yellow]Pipeline interrupted by user[/yellow]")
            raise typer.Exit(130)
        
        except Exception as e:
            try:
                progress.remove_task(task)
            except KeyError:
                pass  # Task already removed or doesn't exist
            rprint(f"[red]Pipeline failed with error: {e}[/red]")
            if verbose:
                console.print_exception()
            raise typer.Exit(1)


@app.command()
def test(
    quick: bool = typer.Option(
        False,
        "--quick",
        help="Run a quick test with minimal data"
    )
):
    """
    Run a test pipeline with a small dataset
    """
    max_songs = 100 if quick else 1000
    
    rprint(f"[blue]Running test pipeline with {max_songs} songs...[/blue]")
    
    result = run_pipeline_sync(
        max_songs=max_songs,
        skip_aoty=quick,  # Skip AOTY for quick tests
        dry_run=True  # Always dry run for tests
    )
    
    _display_results(result)
    
    if result.success:
        rprint("[green]✓ Test completed successfully![/green]")
    else:
        rprint("[red]✗ Test failed![/red]")
        raise typer.Exit(1)


@app.command()
def config():
    """
    Display current configuration
    """
    config_table = Table(title="Current Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_column("Description", style="dim")
    
    configs = [
        ("MAX_SONGS", settings.max_songs, "Maximum songs to process"),
        ("SCRAPE_CONCURRENCY", settings.scrape_concurrency, "AOTY scraping concurrency"),
        ("BATCH_SIZE", settings.batch_size, "Processing batch size"),
        ("DB_BATCH_SIZE", settings.db_batch_size, "Database batch size"),
        ("SPOTIFY_RATE_LIMIT", settings.spotify_rate_limit, "Spotify requests per minute"),
        ("LASTFM_RATE_LIMIT", settings.lastfm_rate_limit, "Last.fm requests per minute"),
        ("AOTY_RATE_LIMIT", settings.aoty_rate_limit, "AOTY requests per minute"),
        ("LOG_LEVEL", settings.log_level, "Logging level"),
    ]
    
    for name, value, description in configs:
        config_table.add_row(name, str(value), description)
    
    console.print(config_table)


@app.command()
def validate():
    """
    Validate environment and dependencies
    """
    rprint("[blue]Validating environment...[/blue]")
    
    validation_table = Table(title="Environment Validation")
    validation_table.add_column("Check", style="cyan")
    validation_table.add_column("Status", style="bold")
    validation_table.add_column("Details", style="dim")
    
    checks = []
    
    # Check API keys
    if settings.lastfm_api_key:
        checks.append(("Last.fm API Key", "✓", "Present"))
    else:
        checks.append(("Last.fm API Key", "✗", "Missing"))
    
    if settings.spotify_client_id and settings.spotify_client_secret:
        checks.append(("Spotify Credentials", "✓", "Present"))
    else:
        checks.append(("Spotify Credentials", "✗", "Missing"))
    
    if settings.supabase_url and settings.supabase_service_role_key:
        checks.append(("Supabase Credentials", "✓", "Present"))
    else:
        checks.append(("Supabase Credentials", "✗", "Missing"))
    
    # Check directories
    import os
    for dir_name, dir_path in [
        ("Data Directory", settings.data_dir),
        ("Logs Directory", settings.logs_dir),
        ("Exports Directory", settings.exports_dir)
    ]:
        if os.path.exists(dir_path):
            checks.append((dir_name, "✓", f"Exists at {dir_path}"))
        else:
            checks.append((dir_name, "✗", f"Missing: {dir_path}"))
    
    # Add checks to table
    for check, status, details in checks:
        color = "green" if status == "✓" else "red"
        validation_table.add_row(check, f"[{color}]{status}[/{color}]", details)
    
    console.print(validation_table)
    
    # Summary
    failed_checks = sum(1 for _, status, _ in checks if status == "✗")
    if failed_checks == 0:
        rprint("[green]✓ All validation checks passed![/green]")
    else:
        rprint(f"[red]✗ {failed_checks} validation checks failed![/red]")
        raise typer.Exit(1)


def _display_results(result):
    """Display pipeline results in a formatted table"""
    
    # Results table
    results_table = Table(title="Pipeline Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")
    
    results_table.add_row("Status", "SUCCESS" if result.success else "FAILED")
    results_table.add_row("Total Processed", str(result.total_processed))
    results_table.add_row("Successful Inserts", str(result.successful_inserts))
    results_table.add_row("Failed Inserts", str(result.failed_inserts))
    results_table.add_row("Processing Time", f"{result.processing_time_seconds:.2f}s")
    
    console.print(results_table)
    
    # Coverage stats if available
    if result.coverage_stats:
        coverage_table = Table(title="Coverage Statistics")
        coverage_table.add_column("Source", style="cyan")
        coverage_table.add_column("Coverage", style="green")
        
        for key, value in result.coverage_stats.items():
            if key.endswith("_pct"):
                source = key.replace("_pct", "").replace("_", " ").title()
                coverage_table.add_row(source, f"{value:.1f}%")
            elif key.endswith("_count"):
                source = key.replace("_count", "").replace("_", " ").title()
                coverage_table.add_row(source, str(value))
        
        console.print(coverage_table)
    
    # Errors if any
    if result.errors:
        rprint("\n[red]Errors encountered:[/red]")
        for i, error in enumerate(result.errors, 1):
            rprint(f"  {i}. {error}")


def main():
    """Main entry point for the CLI"""
    app()


if __name__ == "__main__":
    main()