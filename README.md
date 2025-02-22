# PersonaFA
Descrição do Projeto
Este projeto é um chatbot desenvolvido em Python que simula a personalidade de um indivíduo a partir de uma coleção de textos (.txt). O sistema utiliza técnicas avançadas de processamento de linguagem natural (NLP) para ler e processar arquivos de texto, gerar embeddings e armazenar essas informações em um banco de dados vetorial. A partir desses dados, o chatbot, alimentado pelo modelo gpt-4o-mini, reproduz a personalidade presente nos textos, interagindo com os usuários através de uma interface intuitiva.

Funcionalidades
Leitura e Processamento de Arquivos: Importa uma base de dados composta por arquivos .txt, realizando a extração e a limpeza dos dados.
Geração de Embeddings com OpenAI: Utiliza os métodos de embedding da OpenAI para converter os textos em vetores, capturando suas características semânticas.
Criação de Banco de Vetores com FAISS: Armazena os embeddings em um banco de dados vetorial baseado em FAISS para facilitar a consulta e a recuperação rápida das informações.
Simulação de Personalidade: Através de um prompt de comando, o chatbot utiliza os dados processados para replicar o estilo e a personalidade do autor dos textos.
Interface via Streamlit: O aplicativo roda via Streamlit, proporcionando uma interface web interativa e de fácil utilização para os usuários.
Tecnologias Utilizadas
Python: Linguagem principal de desenvolvimento.
Streamlit: Framework para a criação de interfaces web interativas.
OpenAI: Métodos de embedding para a conversão dos textos em vetores.
FAISS: Biblioteca utilizada para o armazenamento e consulta de vetores.
GPT-4o-mini: Modelo de chatbot que utiliza os dados processados para simular a personalidade desejada.
Como Executar
Pré-requisitos:
Python 3.8 ou superior.
Instalar as dependências listadas no arquivo requirements.txt (por exemplo, usando pip install -r requirements.txt).
Preparação da Base de Dados:
Organize os arquivos .txt em uma pasta designada para que o aplicativo possa acessá-los.
Iniciando o Aplicativo:
Execute o script principal com o comando:
bash
Copiar
Editar
streamlit run main.py
Siga as instruções exibidas na interface do Streamlit para interagir com o chatbot.
Uso e Finalidade
Este projeto é TOTALMENTE para fins acadêmicos e não possui licença. Não é permitida nenhuma forma de comercialização ou utilização que não esteja estritamente relacionada a pesquisas e estudos acadêmicos.

Contribuições
Contribuições são bem-vindas para fins de aprendizado e aprimoramento do projeto. Se você deseja melhorar o projeto ou adicionar novas funcionalidades, sinta-se à vontade para abrir issues ou enviar pull requests, sempre respeitando o propósito acadêmico do mesmo.

Contato
Para dúvidas ou mais informações, abra uma issue neste repositório ou entre em contato diretamente com o(s) mantenedor(es).
