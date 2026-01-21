"""
Einfacher Loader für Prompt-Templates aus Dateien
"""
import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional


class PromptLoader:
    """Lädt Prompt-Templates aus YAML oder TXT Dateien"""
    
    def __init__(self, prompts_dir: str = "prompts"):
        """
        Initialisiert den Prompt-Loader
        
        Args:
            prompts_dir: Verzeichnis mit den Prompt-Dateien
        """
        self.prompts_dir = Path(prompts_dir)
        self.prompts: Dict[str, Dict] = {}
        self._load_all_prompts()
    
    def _load_all_prompts(self):
        """Lädt alle Prompt-Dateien aus dem Verzeichnis"""
        if not self.prompts_dir.exists():
            print(f"Warnung: Prompts-Verzeichnis '{self.prompts_dir}' existiert nicht")
            return
        
        # YAML-Dateien laden
        for yaml_file in self.prompts_dir.glob("*.yml"):
            self._load_yaml_prompt(yaml_file)
        
        for yaml_file in self.prompts_dir.glob("*.yaml"):
            self._load_yaml_prompt(yaml_file)
        
        # TXT-Dateien laden
        for txt_file in self.prompts_dir.glob("*.txt"):
            self._load_txt_prompt(txt_file)
    
    def _load_yaml_prompt(self, filepath: Path):
        """Lädt ein Prompt-Template aus einer YAML-Datei"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                name = data.get('name', filepath.stem)
                self.prompts[name] = {
                    'name': name,
                    'description': data.get('description', ''),
                    'template': data.get('template', ''),
                    'file': str(filepath)
                }
        except Exception as e:
            print(f"Fehler beim Laden von {filepath}: {e}")
    
    def _load_txt_prompt(self, filepath: Path):
        """Lädt ein Prompt-Template aus einer TXT-Datei"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                name = filepath.stem
                self.prompts[name] = {
                    'name': name,
                    'description': f'Prompt aus {filepath.name}',
                    'template': content,
                    'file': str(filepath)
                }
        except Exception as e:
            print(f"Fehler beim Laden von {filepath}: {e}")
    
    def get_prompt(self, name: str) -> Optional[str]:
        """
        Gibt das Prompt-Template zurück
        
        Args:
            name: Name des Prompts
            
        Returns:
            Template-String oder None
        """
        prompt = self.prompts.get(name)
        return prompt['template'] if prompt else None
    
    def get_prompt_info(self, name: str) -> Optional[Dict]:
        """
        Gibt alle Informationen zu einem Prompt zurück
        
        Args:
            name: Name des Prompts
            
        Returns:
            Dict mit name, description, template, file
        """
        return self.prompts.get(name)
    
    def list_prompts(self) -> List[str]:
        """Gibt eine Liste aller verfügbaren Prompt-Namen zurück"""
        return list(self.prompts.keys())
    
    def format_prompt(self, name: str, query: str, context: str, **kwargs) -> str:
        """
        Formatiert ein Prompt-Template mit den gegebenen Werten
        
        Args:
            name: Name des Prompts
            query: Benutzer-Frage
            context: Kontext aus Retrieval
            **kwargs: Zusätzliche Variablen
            
        Returns:
            Formatierter Prompt-String
        """
        template = self.get_prompt(name)
        if not template:
            raise ValueError(f"Prompt '{name}' nicht gefunden")
        
        return template.format(query=query, context=context, **kwargs)
    
    def print_prompt_info(self):
        """Gibt Informationen über alle geladenen Prompts aus"""
        print(f"\nGeladene Prompts aus '{self.prompts_dir}':")
        print("=" * 80)
        for name, info in self.prompts.items():
            print(f"\n[{name}]")
            print(f"  Beschreibung: {info['description']}")
            print(f"  Datei: {info['file']}")
            preview = info['template'][:100].replace('\n', ' ')
            print(f"  Vorschau: {preview}...")


# Beispiel-Verwendung
if __name__ == "__main__":
    loader = PromptLoader("prompts")
    
    # Alle Prompts anzeigen
    loader.print_prompt_info()
    
    # Einen Prompt verwenden
    print("\n" + "=" * 80)
    print("Beispiel: basic_rag formatiert")
    print("=" * 80)
    
    formatted = loader.format_prompt(
        name="basic_rag",
        query="What is RAG?",
        context="RAG stands for Retrieval-Augmented Generation..."
    )
    print(formatted)