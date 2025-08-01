import os
from django.core.management.base import BaseCommand
from django.conf import settings
from AIBased_Interview.rag_service import rag_service

class Command(BaseCommand):
    help = 'Build RAG index from CSV file'

    def add_arguments(self, parser):
        parser.add_argument(
            '--csv-file',
            type=str,
            default='Final_Dataset_FYP.csv',
            help='Path to the CSV file containing Q&A data'
        )

    def handle(self, *args, **options):
        csv_file = options['csv_file']
        csv_path = os.path.join(settings.BASE_DIR, csv_file)
        
        if not os.path.exists(csv_path):
            self.stdout.write(
                self.style.ERROR(f'CSV file not found at: {csv_path}')
            )
            return
        
        self.stdout.write(
            self.style.SUCCESS(f'Starting to build RAG index from: {csv_path}')
        )
        
        try:
            success = rag_service.build_index_from_csv(csv_path)
            
            if success:
                self.stdout.write(
                    self.style.SUCCESS('RAG index built successfully!')
                )
            else:
                self.stdout.write(
                    self.style.ERROR('Failed to build RAG index.')
                )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error building RAG index: {str(e)}')
            ) 