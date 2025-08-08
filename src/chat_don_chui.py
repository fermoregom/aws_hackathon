# requirements.txt
"""
streamlit>=1.28.0
langchain>=0.1.0
langchain-aws>=0.1.0
langchain-community>=0.0.20
boto3>=1.28.0
python-dotenv>=1.0.0
"""

# chatbot_app.py
import streamlit as st
import boto3
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# LangChain imports
from langchain_aws import ChatBedrockConverse
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.output import LLMResult

# Cargar variables de entorno
load_dotenv()

# Configuración de Streamlit
st.set_page_config(
    page_title="🤖 Chatbot Nova Lite",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Callback para mostrar streaming en tiempo real
class StreamlitCallbackHandler(BaseCallbackHandler):
    """Callback handler para mostrar respuestas en streaming"""
    
    def __init__(self, container):
        self.container = container
        self.text = ""
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        self.container.markdown(self.text)

def create_bedrock_llm(region_name="us-east-2", model_id="us.amazon.nova-lite-v1:0"):
    """Crear instancia de Nova Lite a través de Bedrock"""
    return ChatBedrockConverse(
        client=boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            config=boto3.session.Config(
                read_timeout=300,  # 5 minutos
                connect_timeout=60,  # 1 minuto
                retries={'max_attempts': 2}
            )
        ),
        model=model_id,
        max_tokens=7000,
        temperature=0.15,
        top_p=0.9,
        region_name=region_name
    )

class NovaLiteChatbot:
    """Clase principal del chatbot usando AWS Nova Lite"""
    
    def __init__(self, region_name: str = "us-east-2", model_id: str = "us.amazon.nova-lite-v1:0", 
                 memory_size: int = 10, system_prompt: str = None):
        """
        Inicializar el chatbot
        
        Args:
            region_name: Región de AWS
            model_id: ID del modelo Nova Lite
            memory_size: Cantidad de mensajes a recordar
            system_prompt: Prompt del sistema personalizado
        """
        self.region_name = region_name
        self.model_id = model_id
        self.memory_size = memory_size
        
        # Crear instancia del modelo
        self.llm = create_bedrock_llm(region_name, model_id)
        
        # Configurar memoria conversacional
        self.memory = ConversationBufferWindowMemory(
            k=memory_size,
            return_messages=True
        )
        
        # Sistema de prompts
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.prompt_template = self._create_prompt_template()
        
        # Estadísticas
        self.stats = {
            "total_messages": 0,
            "total_tokens_estimated": 0,
            "session_start": datetime.now(),
            "last_interaction": None
        }
    
    def _default_system_prompt(self) -> str:
        """Prompt del sistema por defecto"""
        return """
        Análisis del problema (texto e imagen):

Texto proporcionado: No has especificado detalles del vehículo (marca, modelo, año) ni síntomas adicionales, solo que la foto muestra un indicador de error en el tablero.
Imagen (simulada): Asumo que la foto muestra una luz de advertencia en el tablero (por ejemplo, check engine, batería, aceite, etc.). Para avanzar, necesito más detalles sobre:

¿Qué luz o símbolo específico aparece en el tablero? (Por ejemplo, un motor, una batería, un signo de exclamación, etc.)
¿Hay un código de error visible (por ejemplo, en una pantalla digital)?
Marca, modelo, año y tipo de motor del vehículo.
Síntomas adicionales (por ejemplo, el auto no arranca, pierde potencia, hace ruidos, etc.).
Condiciones en las que ocurre el problema (por ejemplo, al encender, al acelerar, en ralentí).

Preguntas para aclarar:

¿Puedes describir el símbolo o mensaje exacto que muestra la foto del tablero? (Por ejemplo, ¿es una luz de "check engine", batería, ABS, etc.?)
¿Cuál es la marca, modelo y año de tu auto? (Por ejemplo, Toyota Corolla 2018).
¿El auto presenta otros síntomas, como ruidos, vibraciones, o problemas al conducir?
¿Has usado un escáner OBD-II para obtener códigos de error? Si es así, ¿cuáles son?

Identificación preliminar de la pieza:
Sin detalles específicos, no puedo determinar la pieza exacta, pero puedo ofrecer un análisis inicial basado en problemas comunes asociados con luces de advertencia en el tablero:

Luz de "check engine" (motor): Podría indicar un problema con el sistema de encendido (bujías, bobinas), sensores (como el sensor de oxígeno o MAF), o el sistema de combustible (bomba de combustible, inyectores). La pieza dependerá del código de error específico.
Luz de batería: Podría señalar un fallo en la batería, el alternador, o los cables/terminales (especialmente si hay corrosión visible).
Luz de aceite: Puede indicar baja presión de aceite, lo que podría requerir revisar la bomba de aceite o el sensor de presión.
Luz de ABS o frenos: Podría implicar un problema con los sensores de las ruedas, el módulo ABS, o las pastillas de freno.

Determinación del tipo de persona:
Tu descripción es breve y no usa términos técnicos ("la foto corresponde al tablero del auto donde se muestra el indicador del error"), lo que sugiere que podrías ser una persona sin experiencia técnica o un propietario con conocimientos básicos. Para confirmar, ¿has trabajado antes en reparaciones de autos, o es la primera vez que enfrentas este problema? Esto me ayudará a adaptar la explicación.
Recomendaciones preliminares:

Pasos inmediatos:

Si la luz es de "check engine", te recomiendo usar un escáner OBD-II (disponible en talleres o tiendas de autopartes) para obtener el código de error. Esto ayudará a identificar la pieza específica.
Si la luz es de batería u aceite, revisa visualmente la batería (busca corrosión en los terminales) o el nivel de aceite en el motor.


Dónde buscar ayuda:

Lleva el auto a un taller confiable o una tienda de autopartes que ofrezca escaneos gratuitos de códigos de error.
Si prefieres comprar piezas, sitios como AutoZone, Amazon, o talleres especializados son buenas opciones (dependiendo de la pieza confirmada).


Advertencia: Algunas luces (como la de aceite o frenos) pueden indicar problemas graves. Evita conducir el auto hasta confirmar el diagnóstico para prevenir daños mayores.

Confirmación y claridad:
Por favor, proporciona más detalles sobre la imagen (¿qué luz o mensaje aparece?) y el vehículo (marca, modelo, año, síntomas). Esto me permitirá identificar la pieza exacta y explicártelo de forma clara, adaptada a tu nivel de conocimiento. Si no estás seguro de qué significa el indicador, describe el símbolo o compárteme cualquier mensaje visible.
Te voy a pasar una información de los datos en donde puedes encontrar la información de las autopartes

ID,Nombre de Pieza,Marca de Auto,Modelo,Año,Dimensiones,Fabricante,Precio (MXN),Descripción,Compatibilidad Extra,Estado
PZ0003,Radiador,Chevrolet,Aveo,2022,29x3x3 cm,ACDelco,266.21,Diseño optimizado para mejor rendimiento.,"208, Cronos",Nuevo
PZ0005,Sensor de oxígeno,Peugeot,CX-5,2008,22x14x24 cm,Valeo,580.71,Alta durabilidad y eficiencia comprobada.,"Elantra, Aveo",Nuevo
PZ0002,Filtro de aceite,Toyota,Tiggo,2018,15x13x14 cm,Continental,3985.63,Compatible con múltiples modelos y marcas.,"Impreza, Altima, Tiggo",Nuevo

Te voy a proporcionar varios ejemplos de como debes de responder

Question: Mi auto Chevrolet Aveo 2022 está sobrecalentando y necesito arreglarlo pronto.
Answere: Para tu Chevrolet Aveo 2022, el problema de sobrecalentamiento podría estar relacionado con el Radiador.

Aquí tienes una opción compatible:

Pieza: Radiador

Fabricante: ACDelco

Precio: $266.21 MXN

Dimensiones: 29x3x3 cm

Estado: Nuevo

ID: PZ0003

Descripción: Diseño optimizado para mejor rendimiento.
Question: Necesito un sensor de oxígeno para un Peugeot CX-5 2008.
Answere:Aquí tienes una opción de Sensor de oxígeno compatible con tu Peugeot CX-5 2008:

Fabricante: Valeo

Precio: $580.71 MXN

ID: PZ0005

Compatibilidad extra: Elantra, Aveo

Estado: Nuevo

Descripción: Alta durabilidad y eficiencia comprobada.
Question: Tengo un Toyota Tiggo 2018 y está tirando mucho aceite.
Answere:Para tu Toyota Tiggo 2018, encontramos esta pieza compatible:

Pieza: Filtro de aceite

Fabricante: Continental

Precio: $3,985.63 MXN

Dimensiones: 15x13x14 cm

ID: PZ0002

Compatibilidad extra: Impreza, Altima, Tiggo

Estado: Nuevo

"""
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Crear template de prompt con memoria"""
        return ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
    
    def chat(self, user_input: str, callback_handler: Optional[BaseCallbackHandler] = None) -> Dict[str, Any]:
        """
        Procesar un mensaje del usuario
        
        Args:
            user_input: Mensaje del usuario
            callback_handler: Handler para streaming (opcional)
            
        Returns:
            Diccionario con la respuesta y metadata
        """
        try:
            start_time = datetime.now()
            
            # Configurar callbacks
            callbacks = [callback_handler] if callback_handler else []
            
            # Obtener historial de mensajes
            history = self.memory.chat_memory.messages
            
            # Crear el prompt completo
            formatted_prompt = self.prompt_template.format_messages(
                input=user_input,
                history=history
            )
            
            # Llamar al LLM directamente
            response = self.llm.invoke(formatted_prompt, config={"callbacks": callbacks})
            
            # Guardar en memoria
            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(response.content)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Actualizar estadísticas
            self.stats["total_messages"] += 1
            self.stats["total_tokens_estimated"] += len(user_input.split()) + len(response.content.split())
            self.stats["last_interaction"] = end_time
            
            return {
                "response": response.content,
                "processing_time": processing_time,
                "timestamp": end_time.isoformat(),
                "user_input": user_input,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            error_msg = f"Error al procesar mensaje: {str(e)}"
            
            return {
                "response": "Lo siento, hubo un error al procesar tu mensaje. Por favor, intenta de nuevo.",
                "processing_time": 0,
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "success": False,
                "error": error_msg
            }
    
    def get_conversation_history(self) -> List[BaseMessage]:
        """Obtener historial de conversación"""
        return self.memory.chat_memory.messages
    
    def clear_memory(self):
        """Limpiar memoria conversacional"""
        self.memory.clear()
        self.stats["total_messages"] = 0
        self.stats["total_tokens_estimated"] = 0
        self.stats["session_start"] = datetime.now()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de la sesión"""
        current_time = datetime.now()
        session_duration = (current_time - self.stats["session_start"]).total_seconds()
        
        return {
            **self.stats,
            "session_duration_minutes": round(session_duration / 60, 2),
            "avg_tokens_per_message": (
                self.stats["total_tokens_estimated"] / max(self.stats["total_messages"], 1)
            ),
            "model_info": {
                "model_id": self.model_id,
                "region": self.region_name,
                "memory_size": self.memory_size
            }
        }
    
    def export_conversation(self) -> str:
        """Exportar conversación como JSON"""
        history = []
        for message in self.get_conversation_history():
            history.append({
                "type": message.__class__.__name__,
                "content": message.content,
                "timestamp": datetime.now().isoformat()
            })
        
        export_data = {
            "conversation": history,
            "stats": self.get_stats(),
            "export_timestamp": datetime.now().isoformat()
        }
        
        return json.dumps(export_data, indent=2, ensure_ascii=False)

# Funciones de la interfaz Streamlit
def initialize_chatbot() -> NovaLiteChatbot:
    """Inicializar el chatbot con configuración personalizada"""
    
    # Configuración desde sidebar
    with st.sidebar:
        st.header("⚙️ Configuración del Modelo")
        
        region = st.selectbox(
            "Región AWS",
            ["us-east-2", "us-east-1", "us-west-2", "eu-west-1"],
            index=0
        )
        
        model_options = [
            "us.amazon.nova-lite-v1:0",
            "us.amazon.nova-micro-v1:0",
            "us.amazon.nova-pro-v1:0"
        ]
        
        model_id = st.selectbox(
            "Modelo Nova",
            model_options,
            index=0
        )
        
        memory_size = st.slider(
            "Memoria conversacional",
            min_value=5,
            max_value=50,
            value=10,
            help="Cantidad de mensajes anteriores a recordar"
        )
        
        # Prompt del sistema personalizado
        st.header("🎭 Personalización")
        custom_prompt = st.text_area(
            "Prompt del sistema (opcional)",
            placeholder="Personaliza el comportamiento del asistente...",
            height=100
        )
    
    return NovaLiteChatbot(
        region_name=region,
        model_id=model_id,
        memory_size=memory_size,
        system_prompt=custom_prompt if custom_prompt else None
    )

def display_chat_interface(chatbot: NovaLiteChatbot):
    """Mostrar interfaz principal del chat"""
    
    # Título y descripción
    st.title("🤖 Nova Lite Assistant")
    st.markdown("*Chatbot inteligente powered by AWS Bedrock Nova Lite*")
    
    # Verificar credenciales AWS
    if not (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")):
        st.error("❌ Por favor configura tus credenciales AWS en las variables de entorno")
        st.info("Necesitas: AWS_ACCESS_KEY_ID y AWS_SECRET_ACCESS_KEY")
        return
    
    # Inicializar historial en session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Mostrar estadísticas en la sidebar
    with st.sidebar:
        st.header("📊 Estadísticas")
        stats = chatbot.get_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mensajes", stats["total_messages"])
            st.metric("Duración (min)", stats["session_duration_minutes"])
        with col2:
            st.metric("Tokens aprox.", stats["total_tokens_estimated"])
            st.metric("Promedio tokens", round(stats["avg_tokens_per_message"]))
        
        # Botones de control
        st.header("🛠️ Controles")
        
        if st.button("🗑️ Limpiar conversación"):
            chatbot.clear_memory()
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("💾 Exportar conversación"):
            export_data = chatbot.export_conversation()
            st.download_button(
                label="📥 Descargar JSON",
                data=export_data,
                file_name=f"conversacion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Área principal del chat
    st.header("💬 Conversación")
    
    # Container para el historial
    chat_container = st.container()
    
    # Mostrar historial
    with chat_container:
        for i, (user_msg, bot_response, metadata) in enumerate(st.session_state.chat_history):
            # Mensaje del usuario
            with st.chat_message("user"):
                st.write(user_msg)
            
            # Respuesta del bot
            with st.chat_message("assistant"):
                st.write(bot_response)
                
                # Mostrar metadata en expander
                with st.expander("ℹ️ Detalles"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Tiempo:** {metadata.get('processing_time', 0):.2f}s")
                    with col2:
                        st.write(f"**Estado:** {'✅' if metadata.get('success') else '❌'}")
                    with col3:
                        st.write(f"**Timestamp:** {metadata.get('timestamp', '')[:19]}")
                    
                    if not metadata.get('success') and metadata.get('error'):
                        st.error(f"Error: {metadata['error']}")
    
    # Input para nuevo mensaje
    user_input = st.chat_input("Escribe tu mensaje aquí...")
    
    if user_input:
        # Crear container para respuesta en streaming
        response_container = st.empty()
        
        # Mostrar mensaje del usuario inmediatamente
        with st.chat_message("user"):
            st.write(user_input)
        
        # Procesar respuesta con streaming
        with st.chat_message("assistant"):
            # Container para el streaming
            streaming_container = st.empty()
            
            # Crear callback handler para streaming
            callback_handler = StreamlitCallbackHandler(streaming_container)
            
            # Procesar mensaje
            with st.spinner("🤔 Pensando..."):
                result = chatbot.chat(user_input, callback_handler)
            
            # Agregar al historial
            st.session_state.chat_history.append((
                user_input,
                result["response"],
                result
            ))
        
        # Rerun para mostrar la conversación actualizada
        st.rerun()

def display_examples():
    """Mostrar ejemplos de uso"""
    st.header("💡 Ejemplos de conversación")
    
    examples = [
        "Explícame qué es la inteligencia artificial",
        "¿Puedes ayudarme a escribir un email profesional?",
        "¿Cuáles son las mejores prácticas para programar en Python?",
        "Necesito ideas para un proyecto de machine learning",
        "¿Cómo puedo mejorar mi productividad en el trabajo?"
    ]
    
    for example in examples:
        if st.button(f"💬 {example}", key=f"example_{hash(example)}"):
            # Simular input del usuario
            st.session_state.example_input = example

def main():
    """Función principal de la aplicación"""
    
    try:
        # Inicializar chatbot
        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = initialize_chatbot()
        
        # Tabs principales
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "💡 Ejemplos", "📖 Ayuda"])
        
        with tab1:
            display_chat_interface(st.session_state.chatbot)
        
        with tab2:
            display_examples()
        
        with tab3:
            st.header("📖 Guía de uso")
            st.markdown("""
            ### 🚀 Cómo usar el chatbot:
            
            1. **Configuración**: Ajusta el modelo y parámetros en la barra lateral
            2. **Credenciales**: Asegúrate de tener configuradas las variables de entorno AWS
            3. **Conversación**: Escribe tu mensaje en el campo de texto inferior
            4. **Memoria**: El bot recuerda conversaciones anteriores según la configuración
            5. **Exportar**: Puedes descargar el historial de conversación
            
            ### 🔧 Variables de entorno requeridas:
            ```
            AWS_ACCESS_KEY_ID=tu_access_key
            AWS_SECRET_ACCESS_KEY=tu_secret_key
            ```
            
            ### 📊 Características:
            - ✅ Memoria conversacional configurable
            - ✅ Streaming de respuestas en tiempo real
            - ✅ Estadísticas de uso detalladas
            - ✅ Exportación de conversaciones
            - ✅ Múltiples modelos Nova disponibles
            - ✅ Configuración flexible de prompts
            """)
    
    except Exception as e:
        st.error(f"❌ Error al inicializar la aplicación: {str(e)}")
        st.info("Verifica tu configuración de AWS y las credenciales")

if __name__ == "__main__":
    main()
