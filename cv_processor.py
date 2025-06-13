"""
CV Processor Module - Core CV optimization logic
"""

import PyPDF2
import anthropic
import subprocess
import tempfile
import shutil
import json
import os
import re
import asyncio
from pathlib import Path
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class CVProcessor:
    """Handles CV processing and optimization."""
    
    def __init__(self):
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompts from prompts.json file."""
        try:
            prompts_file = Path(__file__).parent / 'prompts.json'
            with open(prompts_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """Extract text content from a PDF file."""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return None
    
    def extract_text_from_tex(self, tex_path: str) -> Optional[str]:
        """Extract readable text content from LaTeX file by removing commands."""
        try:
            with open(tex_path, 'r', encoding='utf-8') as file:
                tex_content = file.read()
            
            # Remove LaTeX comments
            tex_content = re.sub(r'%.*$', '', tex_content, flags=re.MULTILINE)
            
            # Remove common LaTeX commands but keep their content
            # Remove \command{content} -> content
            tex_content = re.sub(r'\\[a-zA-Z]+\*?\{([^{}]*)\}', r'\1', tex_content)
            
            # Remove \command[options]{content} -> content
            tex_content = re.sub(r'\\[a-zA-Z]+\*?\[[^\]]*\]\{([^{}]*)\}', r'\1', tex_content)
            
            # Remove standalone commands
            tex_content = re.sub(r'\\[a-zA-Z]+\*?', ' ', tex_content)
            
            # Remove curly braces
            tex_content = re.sub(r'[{}]', '', tex_content)
            
            # Clean up whitespace
            tex_content = re.sub(r'\s+', ' ', tex_content)
            tex_content = re.sub(r'\n\s*\n', '\n\n', tex_content)
            
            return tex_content.strip()
        
        except Exception as e:
            logger.error(f"Error extracting text from LaTeX: {e}")
            return None
    
    def read_tex_file(self, tex_path: str) -> Optional[str]:
        """Read the raw LaTeX content."""
        try:
            with open(tex_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading LaTeX file: {e}")
            return None
    
    def create_career_analysis_prompt(self, resume_text: str, tex_content: str, 
                                    job_description: str = "", mode: str = "normal") -> str:
        """Create the complete prompt for Claude using the selected mode."""
        # Select prompt based on mode
        if mode == "unhinged":
            template = self.prompts["unhinged"]
        else:
            template = self.prompts["normal"]
        
        # Add job description if provided
        job_description_section = ""
        if job_description.strip():
            job_description_section = f"\nHere is the EXACT job posting they are targeting:\n\n{job_description}"
        else:
            job_description_section = "\n[NOTE: No job posting provided - general optimization will be performed]"
        
        return template.format(
            resume_text=resume_text,
            tex_content=tex_content,
            job_description_section=job_description_section
        )
    
    async def send_to_claude(self, prompt: str) -> Optional[str]:
        """Send the prompt to Claude 4 and get the response."""
        try:
            logger.info("Sending request to Claude 4...")
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=8000,
                    messages=[
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ]
                )
            )
            
            return response.content[0].text
        
        except Exception as e:
            logger.error(f"Error communicating with Claude: {e}")
            return None
    
    def extract_latex_code(self, response: str, section_name: str) -> Optional[str]:
        """Extract LaTeX code from Claude's response."""
        try:
            # Look for LaTeX code blocks - try both formats
            # First try with explicit 'latex' identifier
            pattern_latex = rf"{section_name}.*?```latex\n(.*?)```"
            match = re.search(pattern_latex, response, re.DOTALL | re.IGNORECASE)
            
            if match:
                return match.group(1).strip()
            
            # If no latex identifier, try plain code blocks after the section
            pattern_plain = rf"{section_name}\s*\n\s*```\n(.*?)```"
            match = re.search(pattern_plain, response, re.DOTALL | re.IGNORECASE)
            
            if match:
                content = match.group(1).strip()
                # Basic heuristic: if it looks like LaTeX/resume content, return it
                if any(keyword in content.lower() for keyword in ['\\', 'experience', 'education', 'skills', 'dear hiring']):
                    return content
            
            logger.warning(f"Could not find {section_name} code in response")
            return None
                
        except Exception as e:
            logger.error(f"Error extracting LaTeX code: {e}")
            return None
    
    def compile_latex_to_pdf(self, latex_content: str, output_path: str, filename_prefix: str) -> Optional[str]:
        """Compile LaTeX content to PDF."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                tex_file = temp_path / f"{filename_prefix}.tex"
                
                # Write LaTeX content to file
                with open(tex_file, 'w', encoding='utf-8') as f:
                    f.write(latex_content)
                
                # Compile LaTeX to PDF (run twice for proper references)
                for i in range(2):
                    result = subprocess.run([
                        'pdflatex', 
                        '-interaction=nonstopmode',
                        '-output-directory', str(temp_path),
                        str(tex_file)
                    ], capture_output=True, text=True, cwd=temp_path)
                    
                    if result.returncode != 0:
                        logger.error(f"LaTeX compilation failed:\n{result.stdout}\n{result.stderr}")
                        return None
                
                # Copy PDF to output location
                pdf_file = temp_path / f"{filename_prefix}.pdf"
                if pdf_file.exists():
                    final_path = Path(output_path) / f"{filename_prefix}.pdf"
                    final_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(pdf_file, final_path)
                    logger.info(f"PDF compiled successfully: {final_path}")
                    return str(final_path)
                else:
                    logger.error("PDF file was not generated")
                    return None
                    
        except Exception as e:
            logger.error(f"Error compiling LaTeX: {e}")
            return None
    
    def check_latex_installation(self) -> bool:
        """Check if LaTeX is installed and accessible."""
        try:
            result = subprocess.run(['pdflatex', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("LaTeX installation found")
                return True
            else:
                return False
        except FileNotFoundError:
            return False
    
    async def process_cv(self, file_path: str, job_description: str, 
                        prompt_type: str, user_info: dict) -> Dict[str, Any]:
        """Main processing function for CV optimization."""
        
        # Check LaTeX installation
        if not self.check_latex_installation():
            raise RuntimeError("LaTeX (pdflatex) not found. Please install a LaTeX distribution.")
        
        # Determine file type and extract content
        file_path_obj = Path(file_path)
        
        if file_path_obj.suffix.lower() == '.pdf':
            logger.info(f"Processing PDF file: {file_path}")
            resume_text = self.extract_text_from_pdf(file_path)
            tex_content = "% Original LaTeX source not available - converted from PDF"
        elif file_path_obj.suffix.lower() == '.tex':
            logger.info(f"Processing LaTeX file: {file_path}")
            resume_text = self.extract_text_from_tex(file_path)
            tex_content = self.read_tex_file(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a .pdf or .tex file")
        
        if not resume_text or not tex_content:
            raise ValueError("Failed to extract content from input file")
        
        logger.info(f"Extracted {len(resume_text)} characters of readable text")
        
        # Create the complete prompt
        prompt = self.create_career_analysis_prompt(
            resume_text, tex_content, job_description, prompt_type
        )
        
        # Send to Claude
        response = await self.send_to_claude(prompt)
        
        if not response:
            raise RuntimeError("Failed to get response from Claude")
        
        # Create output directory for this processing session
        output_dir = Path(tempfile.gettempdir()) / "cv_hacker_results" / user_info["user_id"]
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save complete response
        analysis_file = output_dir / f'career_analysis_{prompt_type}.md'
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write(response)
        
        # Extract and compile LaTeX CV
        files_generated = {}
        cv_latex = None
        cover_latex = None
        
        if prompt_type == "unhinged":
            cv_section_name = "ULTRA-OPTIMIZED ONE-PAGE LATEX RESUME"
            cv_filename = "ultra_optimized_cv"
            cover_section_name = "STRATEGIC COVER LETTER"
            cover_filename = "keyword_cover_letter"
        else:
            cv_section_name = "OPTIMIZED RESUME"
            cv_filename = "optimized_cv"
            cover_section_name = "STRATEGIC COVER LETTER"
            cover_filename = "cover_letter"
        
        # Extract CV LaTeX
        cv_latex = self.extract_latex_code(response, cv_section_name)
        if cv_latex:
            # Save LaTeX source
            cv_tex_file = output_dir / f'{cv_filename}.tex'
            with open(cv_tex_file, 'w', encoding='utf-8') as f:
                f.write(cv_latex)
            files_generated['cv_tex'] = str(cv_tex_file)
            
            # Compile to PDF
            cv_pdf = self.compile_latex_to_pdf(cv_latex, output_dir, cv_filename)
            if cv_pdf:
                files_generated['cv_pdf'] = cv_pdf
                logger.info(f"Optimized CV PDF created: {cv_pdf}")
        
        # Extract cover letter LaTeX
        cover_latex = self.extract_latex_code(response, cover_section_name)
        if cover_latex:
            # Save LaTeX source
            cover_tex_file = output_dir / f'{cover_filename}.tex'
            with open(cover_tex_file, 'w', encoding='utf-8') as f:
                f.write(cover_latex)
            files_generated['cover_tex'] = str(cover_tex_file)
            
            # Compile to PDF
            cover_pdf = self.compile_latex_to_pdf(cover_latex, output_dir, cover_filename)
            if cover_pdf:
                files_generated['cover_pdf'] = cover_pdf
                logger.info(f"Cover letter PDF created: {cover_pdf}")
        
        files_generated['analysis'] = str(analysis_file)
        
        return {
            "analysis": response,
            "cv_latex": cv_latex,
            "cover_latex": cover_latex,
            "files": files_generated
        }