import base64, io, json, os
import re
from datetime import datetime
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq.chat_models import ChatGroq
from groq import Groq
import requests
from bs4 import BeautifulSoup


def main():
    load_dotenv(Path(__file__).with_name('.env'))
    st.set_page_config(page_title='AI Assistant', page_icon='🤖', layout='wide')
    ss = st.session_state
    ss.setdefault('messages', []); ss.setdefault('groq_api_key_override', ''); ss.setdefault('seed_prompt', ''); ss.setdefault('feedback_note', ''); ss.setdefault('vision_cache', {})
    feedback_file = Path(__file__).with_name('feedback.jsonl')
    current_words = {'latest', 'current', 'today', 'recent', 'news', 'score', 'squad', 'price', 'weather', 'election', 'match'}
    field_aliases = {
        'candidate name': ['candidate name', 'name of candidate', 'applicant name', 'student name', 'name'],
        'roll number': ['roll number', 'roll no', 'registration number', 'application number'],
        'exam date': ['exam date', 'date of examination', 'date'],
        'venue': ['venue', 'exam centre', 'center', 'centre'],
        'dob': ['date of birth', 'dob', 'birth date'],
    }

    def badge(text, tone='dark'):
        colors = {'dark': '#112032', 'light': '#eff6ff', 'good': '#0f2e24', 'warn': '#3a2a0d'}
        text_colors = {'dark': '#dbeafe', 'light': '#0f172a', 'good': '#bbf7d0', 'warn': '#fde68a'}
        return f"<span style='display:inline-block;padding:6px 10px;border-radius:999px;background:{colors[tone]};color:{text_colors[tone]};border:1px solid #29405c;margin:0 6px 6px 0;font-size:12px'>{text}</span>"

    def overlap(a, b):
        sa = {w for w in a.lower().split() if len(w) > 3}
        sb = {w for w in b.lower().split() if len(w) > 3}
        return len(sa & sb)

    def parse_pdf(blob):
        texts = []
        try:
            from pypdf import PdfReader
            texts += [(p.extract_text() or '') for p in PdfReader(io.BytesIO(blob)).pages]
        except Exception:
            pass
        if not ''.join(texts).strip():
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(blob)) as pdf:
                    texts = [(p.extract_text() or '') for p in pdf.pages]
            except Exception:
                pass
        return '\n'.join(t for t in texts if t).strip()

    def ocr_pdf(blob):
        try:
            import fitz
            import numpy as np
            from rapidocr_onnxruntime import RapidOCR
            ocr, lines = RapidOCR(), []
            doc = fitz.open(stream=blob, filetype='pdf')
            for i in range(min(3, doc.page_count)):
                pix = doc[i].get_pixmap(matrix=fitz.Matrix(2, 2))
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                result, _ = ocr(img)
                if result:
                    lines += [line[1] for line in result]
            return '\n'.join(lines).strip()
        except Exception:
            return ''

    def pdf_images(blob):
        try:
            import fitz
            doc = fitz.open(stream=blob, filetype='pdf')
            imgs = []
            for i in range(min(3, doc.page_count)):
                pix = doc[i].get_pixmap(matrix=fitz.Matrix(2, 2))
                imgs.append('data:image/png;base64,' + base64.b64encode(pix.tobytes('png')).decode())
            return imgs
        except Exception:
            return []

    def vision_pdf_text(name, blob, key):
        if name in ss.vision_cache:
            return ss.vision_cache[name]
        images = pdf_images(blob)
        if not images or not key:
            return ''
        try:
            client = Groq(api_key=key)
            content = [{'type': 'text', 'text': 'Extract all visible key details from this admit card or exam document. Return plain text lines in key: value format. Include candidate name, roll number, application/registration number, exam date, subject, venue, reporting time, and any other visible identity fields.'}]
            content += [{'type': 'image_url', 'image_url': {'url': img}} for img in images[:5]]
            out = client.chat.completions.create(
                model='meta-llama/llama-4-scout-17b-16e-instruct',
                messages=[{'role': 'user', 'content': content}],
                temperature=0,
            )
            text = out.choices[0].message.content or ''
            ss.vision_cache[name] = text
            return text
        except Exception:
            return ''

    def chunk_text(text, size=900, overlap_size=150):
        chunks, i = [], 0
        while i < len(text):
            chunks.append(text[i:i + size])
            i += max(1, size - overlap_size)
        return chunks[:60]

    def field_terms(prompt):
        lower = prompt.lower()
        terms = []
        for _, aliases in field_aliases.items():
            if any(a in lower for a in aliases):
                terms += aliases
        return terms or lower.split()

    def doc_context(prompt, docs):
        if not docs:
            return '', []
        terms = {t.lower() for t in field_terms(prompt) if len(t) > 1}
        ranked = []
        for doc in docs:
            for chunk in doc['chunks']:
                score = sum(t in chunk.lower() for t in terms)
                if re.search(rf"({'|'.join(re.escape(t) for t in list(terms)[:8])})\s*[:\-]", chunk, flags=re.I):
                    score += 3
                if score:
                    ranked.append((score, doc['name'], chunk))
        ranked.sort(key=lambda x: x[0], reverse=True)
        top = ranked[:5]
        ctx = '\n\n'.join(f"Document: {name}\n{chunk}" for _, name, chunk in top)
        return ctx, [name for _, name, _ in top]

    def fetch_page(url):
        try:
            html = requests.get(url, timeout=6, headers={'User-Agent': 'Mozilla/5.0'}).text
            soup = BeautifulSoup(html, 'html.parser')
            for tag in soup(['script', 'style', 'noscript']):
                tag.decompose()
            text = ' '.join(soup.get_text(' ', strip=True).split())
            return text[:5000]
        except Exception:
            return ''

    def search_web(q):
        try:
            from duckduckgo_search import DDGS
            queries = [q, f'{q} today', f'{q} {datetime.now().strftime("%B %d %Y")}']
            rows = []
            with DDGS() as ddgs:
                for query in queries:
                    rows += list(ddgs.news(query, max_results=4) or [])
                    rows += list(ddgs.text(query, max_results=4) or [])
            seen, docs = set(), []
            terms = {w for w in q.lower().split() if len(w) > 2}
            for r in rows:
                url = r.get('url') or r.get('href')
                if not url or url in seen:
                    continue
                seen.add(url)
                title, body = r.get('title', ''), r.get('body', '')
                page = fetch_page(url)
                text = page or body
                score = sum(t in f'{title} {body} {text}'.lower() for t in terms)
                if text:
                    docs.append({'url': url, 'title': title, 'text': text, 'score': score})
            docs = sorted(docs, key=lambda x: x['score'], reverse=True)[:4]
            context = '\n\n'.join(f"Source: {d['title']} | {d['url']}\nContent: {d['text'][:1600]}" for d in docs)
            return context, [d['url'] for d in docs]
        except Exception:
            return '', []

    def save_feedback(entry):
        with feedback_file.open('a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')

    def past_feedback(q):
        if not feedback_file.exists():
            return ''
        picks = []
        for line in feedback_file.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if overlap(q, item.get('prompt', '')) > 0 and (item.get('rating') == 'down' or item.get('correction')):
                picks.append(item)
        return '\n'.join(f"- For similar query `{i['prompt']}` prefer: {i.get('correction') or 'a more accurate and updated answer.'}" for i in picks[-3:])

    with st.sidebar:
        st.title('⚙️ Settings'); st.divider()
        model = st.selectbox('Choose Model', ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant'])
        temp = st.slider('Creativity', 0.0, 1.0, 0.3, 0.1)
        answer_mode = st.selectbox('Answer style', ['Balanced', 'Research', 'Brief'])
        theme = st.radio('Theme', ['Dark', 'Light'], horizontal=True)
        live_search = st.toggle('Live web answers', value=True)
        st.text_input('Groq API Key', key='groq_api_key_override', type='password')
        files = st.file_uploader('Upload file(s)', type=['pdf', 'docx', 'txt', 'csv'], accept_multiple_files=True)
        m1, m2, m3 = st.columns(3)
        m1.metric('Msgs', len(ss.messages))
        m2.metric('Files', len(files or []))
        m3.metric('Web', 'On' if live_search else 'Off')
        if st.button('🗑 Clear Chat', use_container_width=True): ss.messages, ss.seed_prompt = [], ''; st.rerun()
        st.subheader('Chat History')
        for m in ss.messages[-8:]: st.caption(f"{m['role']}: {m['content'][:70]}")

    st.markdown(
        """<style>.stApp{background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);color:#fff!important}.stApp *{color:#fff!important}</style>"""
        if theme == 'Dark' else
        """<style>.stApp{background:linear-gradient(135deg,#f8fbff,#eef5ff,#e6f2ff);color:#0f172a!important}.stApp *{color:#0f172a!important}[data-testid="stSidebar"]{background:#eef5ff!important}.stButton>button,button[kind],button[data-testid],.stDownloadButton>button,[data-testid="stBaseButton-secondary"],[data-testid="stBaseButton-primary"],[data-testid="stFileUploaderDropzone"] button,[data-testid="stFileUploaderDropzone"],[data-testid="stChatInput"]>div,[data-baseweb="input"]>div,[data-baseweb="select"]>div,input,textarea{background:#fff!important;color:#0f172a!important;border:1px solid #cbd5e1!important}</style>""",
        unsafe_allow_html=True,
    )

    st.markdown("<h1 style='margin-bottom:6px'>AI Assistant</h1>", unsafe_allow_html=True)
    st.caption('Live-grounded assistant built with Groq, LangChain, and Streamlit')
    st.markdown(
        badge(f'Model: {model}', 'dark') +
        badge('Live Web On' if live_search else 'Live Web Off', 'good' if live_search else 'warn') +
        badge('File Context Ready' if files else 'No Files', 'light'),
        unsafe_allow_html=True,
    )
    st.write('Try asking:')
    c1, c2, c3 = st.columns(3)
    if c1.button('Explain AI', use_container_width=True): ss.seed_prompt = 'Explain AI in simple terms.'
    if c2.button('Write Python code', use_container_width=True): ss.seed_prompt = 'Write a clean Python function for binary search.'
    if c3.button('Latest headlines', use_container_width=True): ss.seed_prompt = 'What are the latest important headlines today?'

    clean = lambda x: (x or '').strip().strip("'\"")
    key, source = clean(ss.groq_api_key_override), 'sidebar'
    if not key:
        try: key, source = clean(st.secrets.get('GROQ_API_KEY', '')), 'streamlit_secrets'
        except StreamlitSecretNotFoundError: key = ''
    if not key: key, source = clean(os.getenv('GROQ_API_KEY') or os.getenv('GROQ_API_TOKEN') or os.getenv('GROQ_KEY')), 'environment'
    if not key: st.info('Add Groq key in sidebar or set GROQ_API_KEY in .env'); st.stop()
    if not key.startswith('gsk_'): st.error('Invalid key format. Groq key must start with gsk_'); st.stop()
    st.caption(f"Using {source} key: `{key[:4]}...{key[-4:] if len(key) > 8 else '****'}`")

    try: llm = ChatGroq(model=model, groq_api_key=key, temperature=temp)
    except TypeError: llm = ChatGroq(model=model, api_key=key, temperature=temp)

    docs = []
    for f in files or []:
        n, b = f.name.lower(), f.read()
        try:
            if n.endswith('.txt'):
                text = b.decode('utf-8', errors='ignore')
            elif n.endswith('.csv'):
                try:
                    import pandas as pd
                    text = pd.read_csv(io.BytesIO(b)).head(100).to_csv(index=False)
                except Exception:
                    text = b.decode('utf-8', errors='ignore')
            elif n.endswith('.pdf'):
                try:
                    text = parse_pdf(b)
                    if len(text.strip()) < 300:
                        text = ocr_pdf(b) or text
                except Exception: st.warning(f'Cannot parse PDF: {f.name} (install pypdf)')
            elif n.endswith('.docx'):
                try:
                    import docx2txt
                    text = docx2txt.process(io.BytesIO(b)) or ''
                except Exception: st.warning(f'Cannot parse DOCX: {f.name} (install docx2txt)')
            else:
                text = ''
            if n.endswith('.pdf') and len(text.strip()) < 300:
                vision = vision_pdf_text(f.name, b, key)
                text = f'{text}\n\n{vision}'.strip()
            if text:
                docs.append({'name': f.name, 'text': text, 'chunks': chunk_text(text)})
        except Exception as e: st.warning(f'File error {f.name}: {e}')
    file_context = '\n\n'.join(d['text'][:2000] for d in docs).strip()

    for m in ss.messages:
        with st.chat_message(m['role'], avatar='👤' if m['role'] == 'user' else '🤖'): st.write(m['content'])

    prompt = st.chat_input('Ask me anything...') or ss.seed_prompt; ss.seed_prompt = ''
    if not prompt:
        dump = '\n'.join(f"{m['role']}: {m['content']}" for m in ss.messages)
        st.download_button('Download Chat', dump or 'No chat yet', file_name='chat_history.txt', use_container_width=True)
        st.markdown('---'); st.caption('Built using LangChain, Groq, and Streamlit'); return

    ss.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user', avatar='👤'): st.write(prompt)

    rag, doc_hits = doc_context(prompt, docs)

    web_ctx, sources = ('', [])
    if live_search and (any(w in prompt.lower() for w in current_words) or len(prompt.split()) > 5):
        web_ctx, sources = search_web(prompt)
    fb = past_feedback(prompt)
    history = '\n'.join(f"{m['role']}: {m['content']}" for m in ss.messages[-10:-1])
    chain = ChatPromptTemplate.from_messages([
        ('system', 'You are a concise, accurate, and helpful AI assistant. Today is {today}.'),
        ('system', 'For latest facts, answer from live web context first. If the exact value or fact is present, state it directly. If the sources do not contain the exact answer, say that clearly and do not invent one.\n\nLive web context:\n{web_ctx}'),
        ('system', 'For uploaded documents, answer from document context first. If a field like candidate name, roll number, date, venue, or ID is explicitly present, return that exact value. If not present, say it is not found in the uploaded document.\n\nDocument context:\n{rag}'),
        ('system', 'Use these lessons from past user feedback when relevant:\n{fb}'),
        ('system', 'Answer style: {answer_style}. Balanced = useful and concise. Research = detailed with source-driven reasoning. Brief = shortest correct answer.'),
        ('human', 'Conversation:\n{history}\n\nUser question: {prompt}\n\nReturn the best current answer. If using live context, prefer exact numbers, names, dates, and sources from that context.')
    ]) | llm

    with st.chat_message('assistant', avatar='🤖'):
        with st.spinner('🤖 Thinking...'):
            resp, box = '', st.empty()
            try:
                data = {'today': datetime.now().strftime('%Y-%m-%d'), 'web_ctx': web_ctx or 'No live web context found.', 'rag': rag or 'No uploaded file context.', 'fb': fb or 'No prior feedback.', 'history': history or 'No prior conversation.', 'prompt': prompt, 'answer_style': answer_mode}
                for ch in chain.stream(data): resp += getattr(ch, 'content', '') or ''; box.write(resp)
                if not resp.strip(): resp = getattr(chain.invoke(data), 'content', '') or 'No response.'; box.write(resp)
            except Exception as e: resp = f'Groq API error: {e}'; box.write(resp)
    ss.messages.append({'role': 'assistant', 'content': resp})
    st.markdown(
        badge('Web-grounded' if sources else 'Model-only', 'good' if sources else 'warn') +
        badge(f'Doc hits: {len(doc_hits)}' if rag else 'No doc context', 'light') +
        badge('Feedback-aware' if fb else 'No feedback match', 'dark'),
        unsafe_allow_html=True,
    )
    answer_tab, source_tab, improve_tab = st.tabs(['Answer Meta', 'Sources', 'Improve'])
    with answer_tab:
        a1, a2, a3 = st.columns(3)
        a1.metric('Sources', len(sources))
        a2.metric('Answer chars', len(resp))
        a3.metric('Mode', answer_mode)
    with source_tab:
        if sources:
            for i, src in enumerate(sources, 1):
                st.markdown(f'**{i}.** {src}')
        else:
            st.info('No live web sources were attached to this answer.')
        if doc_hits:
            st.markdown('**Document matches**')
            for i, name in enumerate(doc_hits, 1):
                st.markdown(f'**{i}.** {name}')
    with improve_tab:
        ss.feedback_note = st.text_area('Correction or preferred answer', value=ss.feedback_note, key='feedback_note_box')
        c1, c2 = st.columns(2)
        if c1.button('👍 Good answer', use_container_width=True):
            save_feedback({'prompt': prompt, 'response': resp, 'rating': 'up', 'correction': ''})
            ss.feedback_note = ''
            st.success('Feedback saved.')
        if c2.button('👎 Needs improvement', use_container_width=True):
            entry = {'prompt': prompt, 'response': resp, 'rating': 'down', 'correction': ss.feedback_note.strip()}
            save_feedback(entry)
            ss.feedback_note = ''
            st.success('Feedback saved and will be reused for similar queries.')

    dump = '\n'.join(f"{m['role']}: {m['content']}" for m in ss.messages)
    st.download_button('Download Chat', dump, file_name='chat_history.txt', use_container_width=True)
    st.markdown('---'); st.caption('Built using LangChain, Groq, and Streamlit')


if __name__ == '__main__':
    main()
