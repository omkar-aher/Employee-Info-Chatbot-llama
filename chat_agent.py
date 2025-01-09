import json
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

class EmployeeChatAgent:
    def __init__(self):
        try:
            # Load employee data
            with open('data/employees.json', 'r') as f:
                self.employee_data = json.load(f)
            
            # Initialize Ollama with llama2
            self.llm = Ollama(
                model="llama2",
                base_url="http://localhost:11434"
            )
            
            # Test the model
            print("Testing LLM connection...")
            test_response = self.llm("Say 'OK' if you can hear me")
            print(f"LLM Test Response: {test_response}")
            
        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise
        
        # Create prompt template with specific instructions
        self.prompt_template = PromptTemplate(
            input_variables=["question", "employee_data"],
            template="""
            You are an HR assistant. Answer questions about employee data.
            
            Employee Data:
            {employee_data}
            
            Question: {question}
            
            Rules:
            1. Only use information from the provided employee data
            2. If information is not available, say so
            3. Keep answers brief and to the point
            4. For skills and certifications, list them clearly
            
            Answer:"""
        )

    def get_response(self, question):
        try:
            # Format prompt with current question and employee data
            prompt = self.prompt_template.format(
                question=question,
                employee_data=json.dumps(self.employee_data, indent=2)
            )
            
            print(f"Sending prompt to LLM: {prompt[:100]}...")  # Print first 100 chars of prompt
            
            # Get response from model
            response = self.llm(prompt)
            
            print(f"Received response: {response[:100]}...")  # Print first 100 chars of response
            
            return response
            
        except Exception as e:
            error_message = f"Error getting response: {str(e)}"
            print(error_message)
            return error_message 