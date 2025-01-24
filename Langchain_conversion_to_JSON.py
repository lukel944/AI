from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel, ValidationError
from typing import Dict
import os
import json

# === MODELE PYDANTIC ===

class TableData(BaseModel):
    table_name: str
    rows: list[dict]

class FinancialData(BaseModel):
    net_asset_value: Dict[str, str]
    isin_code: Dict[str, str]
    tables: list[TableData]

# Funkcja do walidacji JSON za pomocą Pydantic
def validate_financial_data(json_data: dict) -> FinancialData:
    """
    Waliduje dane JSON zgodnie z modelem Pydantic FinancialData.
    """
    try:
        return FinancialData(**json_data)
    except ValidationError as e:
        raise ValueError(f"Validation failed: {e}")

# Funkcja do oczyszczania zbędnego tekstu JSON i konwersji na JSON
def extract_json_from_output(output: str) -> dict:
    """
    Ekstrakcja czystego JSON z tekstowego wyjścia.
    """
    try:
        json_start = output.find("{")
        json_end = output.rfind("}") + 1
        json_str = output[json_start:json_end]
        return json.loads(json_str)
    except (ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to extract JSON: {e}")

# Główna funkcja LLM
def process_text(ocr_text: str, json_schema: dict, example_json: dict, model_name: str = "gpt-4o-mini") -> dict:
    """
    Przetwarza tekst markdown na strukturalny JSON za pomocą modelu LLM.
    """

    # Przygotowanie prompta
    schema_str = json.dumps(json_schema, indent=4)
    example_str = json.dumps(example_json, indent=4)
    template = """
        You are an expert in financial data extraction. Convert the following text into a structured JSON format:
        Text: {ocr_text}

        JSON Schema:
        {{schema_str}}

        Example JSON:
        {{example_str}}

        Please ensure the output strictly matches the schema.
        Instructions:
        - Extract all information, including `net_asset_value`, `isin_code`, and any tables present in the text.
        - Tables should be represented as objects with `table_name` (the title of the table) and `rows` (a list of rows).
        - Each row in `rows` should be a dictionary where keys are column headers.
        - Preserve all numerical values, percentages, and dates exactly as they appear in the text.
        - If the input text does not include specific parts of the schema, include them as empty structures in the JSON.

        Do not include any additional text, explanations, or formatting outside of the JSON object.
        Return only the JSON object.
        """
    prompt = PromptTemplate(input_variables=["ocr_text", "schema_str", "example_str"], template=template)

    # Konfiguracja modelu
    llm = ChatOpenAI(model=model_name, temperature=0.0, openai_api_key=os.environ.get("OPENAI_API_KEY"))
    chain = LLMChain(llm=llm, prompt=prompt)

    # Wywołanie modelu
    output = chain.run(ocr_text=ocr_text, schema_str=schema_str, example_str=example_str)

    # Ekstrakcja i walidacja JSON
    clean_json = extract_json_from_output(output)
    validated_data = validate_financial_data(clean_json)

    return validated_data.model_dump()