import os
import json
import boto3
import logging
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from prompt_awss_hack import prompt_pieza
from typing import List, Dict, Any, Optional
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationMemory:
    """Maneja la memoria de conversación"""
    
    def __init__(self, max_messages: int = 10):
        self.messages: List[Dict[str, Any]] = []
        self.max_messages = max_messages
    
    def add_message(self, role: str, content: str, timestamp: Optional[str] = None):
        """Añade un mensaje a la memoria"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp
        }
        
        self.messages.append(message)
        
        # Mantener solo los últimos max_messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Retorna el historial de conversación"""
        return self.messages.copy()
    
    def clear_memory(self):
        """Limpia la memoria de conversación"""
        self.messages = []

class LocalCSVSearcher:
    """Maneja la búsqueda en archivos CSV almacenados localmente en la carpeta assets"""
    
    def __init__(self, assets_path: str = 'assets'):
        self.assets_path = assets_path
        self.cached_dataframes = {}
    
    def load_csv_from_local(self, file_name: str) -> pd.DataFrame:
        """Carga un archivo CSV desde la carpeta local assets"""
        try:
            # Crear la ruta completa al archivo
            file_path = os.path.join(self.assets_path, file_name)
            
            # Verificar si ya está en caché
            if file_name in self.cached_dataframes:
                logger.info(f"Usando CSV en caché: {file_name}")
                return self.cached_dataframes[file_name]
            
            logger.info(f"Cargando CSV desde: {file_path}")
            
            # Verificar si el archivo existe
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"El archivo {file_path} no se encuentra en la carpeta {self.assets_path}")
            
            # Leer el CSV
            df = pd.read_csv(file_path)
            
            # Guardar en caché
            self.cached_dataframes[file_name] = df
            
            logger.info(f"CSV cargado exitosamente. Filas: {len(df)}, Columnas: {len(df.columns)}")
            return df
            
        except Exception as e:
            logger.error(f"Error al cargar CSV desde local: {str(e)}")
            raise
    
    def search_piece(self, file_name: str, piece_identifier: str, search_columns: List[str] = None) -> Dict[str, Any]:
        """
        Busca una pieza específica en el CSV
        
        Args:
            file_name: Nombre del archivo en la carpeta assets
            piece_identifier: Identificador de la pieza a buscar
            search_columns: Columnas donde buscar (si es None, busca en todas)
        
        Returns:
            Diccionario con los resultados de la búsqueda
        """
        try:
            df = self.load_csv_from_local(file_name)
            
            if search_columns is None:
                search_columns = df.columns.tolist()
            
            # Buscar la pieza
            results = []
            for column in search_columns:
                if column in df.columns:
                    # Búsqueda exacta
                    exact_matches = df[df[column].astype(str).str.upper() == piece_identifier.upper()]
                    
                    # Búsqueda parcial (contiene)
                    partial_matches = df[df[column].astype(str).str.contains(piece_identifier, case=False, na=False)]
                    
                    for _, row in exact_matches.iterrows():
                        results.append({
                            'match_type': 'exact',
                            'matched_column': column,
                            'matched_value': str(row[column]),
                            'row_data': row.to_dict()
                        })
                    
                    # Solo agregar coincidencias parciales si no hay exactas
                    if exact_matches.empty:
                        for _, row in partial_matches.iterrows():
                            results.append({
                                'match_type': 'partial',
                                'matched_column': column,
                                'matched_value': str(row[column]),
                                'row_data': row.to_dict()
                            })
            
            # Remover duplicados
            seen = set()
            unique_results = []
            for result in results:
                # Usar el índice de la fila como identificador único
                row_id = str(sorted(result['row_data'].items()))
                if row_id not in seen:
                    seen.add(row_id)
                    unique_results.append(result)
            
            return {
                'piece_identifier': piece_identifier,
                'total_matches': len(unique_results),
                'results': unique_results,
                'search_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en búsqueda de pieza: {str(e)}")
            return {
                'piece_identifier': piece_identifier,
                'total_matches': 0,
                'results': [],
                'error': str(e),
                'search_timestamp': datetime.now().isoformat()
            }

class NovaProChatbot:
    """Chatbot principal usando AWS Nova Pro con memoria y búsqueda CSV"""
    
    def __init__(self, csv_file_name: str, aws_region: str = 'us-east-1', assets_path: str = '../'):
        self.bedrock_client = ChatBedrockConverse(
            client=boto3.client(
                service_name='bedrock-runtime',
                region_name=aws_region,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                config=boto3.session.Config(
                    read_timeout=300,  # 5 minutos
                    connect_timeout=60,  # 1 minuto
                    retries={'max_attempts': 2})
            ),
            model="amazon.nova-pro-v1:0",
            max_tokens=7000,
            temperature=0.15,
            top_p=0.9,
            region_name=aws_region
        )
        self.memory = ConversationMemory()
        self.csv_searcher = LocalCSVSearcher(assets_path)
        self.csv_file_name = "base_autopartes_dummy.csv"
        self.model_id = "amazon.nova-pro-v1:0"
    
    def format_search_results(self, search_results: Dict[str, Any]) -> str:
        """Formatea los resultados de búsqueda para el contexto del modelo"""
        if search_results['total_matches'] == 0:
            return f"No se encontraron resultados para la pieza: {search_results['piece_identifier']}"
        
        formatted_text = f"Resultados de búsqueda para '{search_results['piece_identifier']}' ({search_results['total_matches']} coincidencias):\n\n"
        
        for i, result in enumerate(search_results['results'][:5], 1):  # Limitar a 5 resultados
            formatted_text += f"Resultado {i} ({result['match_type']} match en {result['matched_column']}):\n"
            for key, value in result['row_data'].items():
                formatted_text += f"  {key}: {value}\n"
            formatted_text += "\n"
        
        return formatted_text
    
    def call_nova_pro(self, user_message: str, search_context: str = "") -> str:
        """Llama al modelo Nova Pro con el contexto completo"""
        try:
            # Construir el contexto de la conversación
            conversation_context = ""
            for msg in self.memory.get_conversation_history():
                conversation_context += f"{msg['role']}: {msg['content']}\n"
            
            # Construir el prompt completo
            system_prompt = """Eres un asistente especializado en ayudar con consultas sobre piezas y componentes. 
            Tienes acceso a una base de datos de piezas y puedes recordar conversaciones anteriores.
            
            Cuando el usuario mencione una pieza específica, búscala automáticamente y proporciona información relevante.
            Sé conciso pero informativo en tus respuestas."""
            prompt_pieza
            full_prompt = prompt_pieza.format_map({"user_message": user_message})

            # Preparar el cuerpo de la solicitud
            request_body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": full_prompt
                            }
                        ]
                    }
                ],
                "inferenceConfig": {
                    "max_new_tokens": 1000,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            
            # Llamar al modelo
            response = self.bedrock_client.invoke([HumanMessage(content=full_prompt)])
            
            # self.bedrock_client.converse(
            #     modelId=self.model_id,
            #     messages=request_body["messages"],
            #     inferenceConfig=request_body["inferenceConfig"]
            # )
            
            # Extraer la respuesta
            assistant_response = response['output']['message']['content'][0]['text']
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"Error al llamar Nova Pro: {str(e)}")
            return f"Error al procesar la consulta: {str(e)}"
    
    def detect_piece_query(self, message: str) -> Optional[str]:
        """Detecta si el mensaje contiene una consulta sobre una pieza específica"""
        # Palabras clave que indican búsqueda de pieza
        keywords = ['pieza', 'parte', 'componente', 'buscar', 'encontrar', 'número', 'código']
        
        message_lower = message.lower()
        
        # Si contiene palabras clave, intentar extraer identificador
        if any(keyword in message_lower for keyword in keywords):
            # Buscar patrones comunes de identificadores
            import re
            
            # Patrones comunes para códigos de pieza
            patterns = [
                r'\b[A-Z0-9]+-[A-Z0-9]+\b',  # ABC-123
                r'\b[A-Z]{2,4}\d{3,6}\b',     # ABC123, ABCD1234
                r'\b\d{4,8}\b',               # 123456
                r'\b[A-Z]\d{3,6}\b'           # A123456
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, message.upper())
                if matches:
                    return matches[0]
        
        return None
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """Función principal de chat"""
        try:
            # Añadir mensaje del usuario a la memoria
            self.memory.add_message("user", user_message)
            
            # Detectar si es una consulta de pieza
            piece_id = self.detect_piece_query(user_message)
            search_context = ""
            search_results = None
            
            if piece_id:
                logger.info(f"Detectada consulta de pieza: {piece_id}")
                search_results = self.csv_searcher.search_piece(self.csv_file_name, piece_id)
                search_context = f"\nInformación de la base de datos:\n{self.format_search_results(search_results)}"
            
            # Generar respuesta usando Nova Pro
            assistant_response = self.call_nova_pro(user_message, search_context)
            
            # Añadir respuesta a la memoria
            self.memory.add_message("assistant", assistant_response)
            
            return {
                "response": assistant_response,
                "search_performed": piece_id is not None,
                "piece_searched": piece_id,
                "search_results": search_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en chat: {str(e)}")
            return {
                "response": f"Error en el sistema: {str(e)}",
                "search_performed": False,
                "piece_searched": None,
                "search_results": None,
                "timestamp": datetime.now().isoformat()
            }

# Función de ejemplo de uso
def main():
    """Ejemplo de uso del sistema"""
    
    # Configuración
    CSV_FILE_NAME = "pieces.csv"  # Nombre del archivo en la carpeta assets
    ASSETS_PATH = "assets"  # Ruta a la carpeta assets
    
    # Crear instancia del chatbot
    chatbot = NovaProChatbot(CSV_FILE_NAME, assets_path=ASSETS_PATH)
    
    # Ejemplo de conversación
    messages = [
        "Hola, necesito ayuda con unas piezas",
        "Busca la pieza ABC-123",
        "¿Qué características tiene esa pieza?",
        "Busca ahora la pieza XYZ-456"
    ]
    
    print("=== Iniciando conversación con Nova Pro ===\n")
    
    for message in messages:
        print(f"Usuario: {message}")
        
        result = chatbot.chat(message)
        
        print(f"Asistente: {result['response']}")
        
        if result['search_performed']:
            print(f"[Búsqueda realizada para: {result['piece_searched']}]")
            if result['search_results'] and result['search_results']['total_matches'] > 0:
                print(f"[{result['search_results']['total_matches']} coincidencias encontradas]")
        
        print("-" * 50)

if __name__ == "__main__":
    main()