import re
import math
import time
import threading
import queue
from collections import defaultdict, Counter, deque
from urllib.parse import urljoin, urlparse, urlunparse
from html.parser import HTMLParser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs
from urllib.parse import urlparse

import urllib.request
import socket
from urllib.error import URLError, HTTPError
import ssl
import os
# HTMLパーサー
class ContentParser(HTMLParser):
    def __init__(self, base_url):
        super().__init__()
        self.base_url = base_url
        self.title = ""
        self.content = []
        self.links = []
        self.in_title = False
        self.in_script = False
        self.in_style = False
        self.meta_description = ""
    
    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self.in_title = True
        elif tag == "script":
            self.in_script = True
        elif tag == "style":
            self.in_style = True
        elif tag == "a":
            for attr, val in attrs:
                if attr == "href":
                    # 相対URLを絶対URLに変換
                    absolute_url = urljoin(self.base_url, val)
                    self.links.append(absolute_url)
        elif tag == "meta":
            attrs_dict = dict(attrs)
            if attrs_dict.get("name") == "description":
                self.meta_description = attrs_dict.get("content", "")
    
    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        elif tag == "script":
            self.in_script = False
        elif tag == "style":
            self.in_style = False
    
    def handle_data(self, data):
        if self.in_script or self.in_style:
            return
        if self.in_title:
            self.title += data
        else:
            text = data.strip()
            if text:
                self.content.append(text)

# URLの正規化
def normalize_url(url):
    """URLを正規化（フラグメント削除、小文字化など）"""
    parsed = urlparse(url)
    # フラグメント削除、スキームとネットロケーションを小文字化
    normalized = urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        parsed.path,
        parsed.params,
        parsed.query,
        ""  # fragment削除
    ))
    return normalized

# robots.txt パーサー
class RobotsTxtParser:
    def __init__(self):
        self.disallowed = {}
        self.crawl_delay = {}
    
    def fetch_robots_txt(self, domain):
        """robots.txtを取得"""
        try:
            robots_url = f"https://{domain}/robots.txt"
            req = urllib.request.Request(robots_url, headers={'User-Agent': 'CustomSearchBot/1.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                content = response.read().decode('utf-8', errors='ignore')
                self.parse_robots_txt(domain, content)
        except:
            pass
    
    def parse_robots_txt(self, domain, content):
        """robots.txtを解析"""
        self.disallowed[domain] = []
        current_agent = None
        
        for line in content.split('\n'):
            line = line.split('#')[0].strip()
            if not line:
                continue
            
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue
            
            directive = parts[0].strip().lower()
            value = parts[1].strip()
            
            if directive == 'user-agent':
                current_agent = value
            elif directive == 'disallow' and current_agent == '*':
                if value:
                    self.disallowed[domain].append(value)
            elif directive == 'crawl-delay' and current_agent == '*':
                try:
                    self.crawl_delay[domain] = float(value)
                except:
                    pass
    
    def can_fetch(self, url):
        """URLがクロール可能か確認"""
        parsed = urlparse(url)
        domain = parsed.netloc
        
        if domain not in self.disallowed:
            return True
        
        path = parsed.path
        for disallowed_path in self.disallowed[domain]:
            if path.startswith(disallowed_path):
                return False
        
        return True
    
    def get_crawl_delay(self, domain):
        """クロール遅延を取得"""
        return self.crawl_delay.get(domain, 1.0)

# リアルタイムWebクローラー
class RealTimeWebCrawler:
    def __init__(self, max_pages=1000, max_threads=10, max_depth=3):
        self.max_pages = max_pages
        self.max_threads = max_threads
        self.max_depth = max_depth
        self.visited = set()
        self.pages = {}
        self.url_queue = queue.Queue()
        self.lock = threading.Lock()
        self.active_threads = 0
        self.robots_parser = RobotsTxtParser()
        self.domain_last_access = {}
        self.running = True
        
        # SSL証明書検証を無効化（本番では推奨されない）
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
    def fetch_page(self, url, timeout=10):
        """Webページを取得"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; CustomSearchBot/1.0)',
                'Accept': 'text/html,application/xhtml+xml',
                'Accept-Language': 'ja,en;q=0.9',
            }
            
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout, context=self.ssl_context) as response:
                if response.status == 200:
                    content_type = response.headers.get('Content-Type', '')
                    if 'text/html' in content_type:
                        content = response.read()
                        encoding = response.headers.get_content_charset('utf-8')
                        return content.decode(encoding, errors='ignore')
        except (URLError, HTTPError, socket.timeout, UnicodeDecodeError) as e:
            print(f"エラー: {url} - {str(e)[:50]}")
        except Exception as e:
            print(f"予期しないエラー: {url} - {str(e)[:50]}")
        
        return None
    
    def parse_page(self, url, html):
        """ページを解析"""
        parser = ContentParser(url)
        try:
            parser.feed(html)
        except:
            return None, []
        
        title = parser.title.strip() or "無題"
        content = " ".join(parser.content)
        links = parser.links
        
        # メタディスクリプションがあれば使用
        if parser.meta_description:
            content = parser.meta_description + " " + content
        
        return {
            "title": title[:200],  # タイトルを制限
            "content": content[:2000],  # コンテンツを制限
            "links": links
        }, links
    
    def should_crawl_domain(self, url):
        """ドメインをクロールすべきか判定"""
        parsed = urlparse(url)
        domain = parsed.netloc
        
        # robots.txtチェック
        if not self.robots_parser.can_fetch(url):
            return False
        
        # クロール遅延の確認
        current_time = time.time()
        if domain in self.domain_last_access:
            delay = self.robots_parser.get_crawl_delay(domain)
            time_since_last = current_time - self.domain_last_access[domain]
            if time_since_last < delay:
                time.sleep(delay - time_since_last)
        
        self.domain_last_access[domain] = time.time()
        return True
    
    def is_valid_url(self, url):
        """有効なURLか確認"""
        try:
            parsed = urlparse(url)
            
            # スキームチェック
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # ネットロケーションチェック
            if not parsed.netloc:
                return False
            
            # 除外パターン
            excluded_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.exe', '.mp4', '.mp3']
            if any(parsed.path.lower().endswith(ext) for ext in excluded_extensions):
                return False
            
            return True
        except:
            return False
    
    def crawl_worker(self):
        """クローラーワーカースレッド"""
        with self.lock:
            self.active_threads += 1
        
        try:
            while self.running and len(self.pages) < self.max_pages:
                try:
                    url, depth = self.url_queue.get(timeout=2)
                except queue.Empty:
                    break
                
                # 正規化
                url = normalize_url(url)
                
                # 訪問済みチェック
                with self.lock:
                    if url in self.visited or len(self.pages) >= self.max_pages:
                        self.url_queue.task_done()
                        continue
                    self.visited.add(url)
                
                print(f"クロール中: {url} (深さ: {depth}, 総ページ数: {len(self.pages)})")
                
                # ドメインクロール可否チェック
                if not self.should_crawl_domain(url):
                    self.url_queue.task_done()
                    continue
                
                # ページ取得
                html = self.fetch_page(url)
                if not html:
                    self.url_queue.task_done()
                    continue
                
                # ページ解析
                page_data, links = self.parse_page(url, html)
                if not page_data:
                    self.url_queue.task_done()
                    continue
                
                # ページを保存
                with self.lock:
                    self.pages[url] = page_data
                
                # 新しいリンクをキューに追加（深さ制限内）
                if depth < self.max_depth:
                    for link in links[:20]:  # リンク数を制限
                        if self.is_valid_url(link):
                            normalized_link = normalize_url(link)
                            with self.lock:
                                if normalized_link not in self.visited:
                                    self.url_queue.put((normalized_link, depth + 1))
                
                self.url_queue.task_done()
                
                # 短い待機
                time.sleep(0.5)
        
        finally:
            with self.lock:
                self.active_threads -= 1
    
    def crawl(self, seed_urls):
        """クローリング開始"""
        print("=" * 60)
        print("リアルタイムWebクローラー起動")
        print(f"最大ページ数: {self.max_pages}, スレッド数: {self.max_threads}, 最大深さ: {self.max_depth}")
        print("=" * 60)
        
        # シードURLをキューに追加
        for url in seed_urls:
            if self.is_valid_url(url):
                parsed = urlparse(url)
                domain = parsed.netloc
                self.robots_parser.fetch_robots_txt(domain)
                self.url_queue.put((url, 0))
        
        # ワーカースレッドを起動
        threads = []
        for _ in range(self.max_threads):
            t = threading.Thread(target=self.crawl_worker)
            t.daemon = True
            t.start()
            threads.append(t)
        
        # クローリング進捗表示
        try:
            while self.running:
                time.sleep(3)
                with self.lock:
                    pages_count = len(self.pages)
                    queue_size = self.url_queue.qsize()
                    active = self.active_threads
                
                print(f"進捗: {pages_count}/{self.max_pages}ページ, キュー: {queue_size}, アクティブ: {active}")
                
                if pages_count >= self.max_pages or (queue_size == 0 and active == 0):
                    break
        except KeyboardInterrupt:
            print("\n\nクローリング中断...")
            self.running = False
        
        # 全スレッド終了待機
        for t in threads:
            t.join(timeout=5)
        
        print(f"\nクローリング完了: {len(self.pages)}ページ")
        return self.pages

# インデックス処理（前回と同じ）
class SearchIndex:
    def __init__(self):
        self.inverted_index = defaultdict(list)
        self.doc_vectors = {}
        self.documents = {}
        self.idf = {}
        self.page_rank = {}
        
    def tokenize(self, text):
        """日本語・英語対応トークナイザー"""
        text = text.lower()
        tokens = re.findall(r'[a-z0-9]+|[ぁ-んァ-ン一-龥々]+', text)
        return tokens
    
    def build_index(self, pages):
        """転置インデックスとTF-IDFを構築"""
        print("\nインデックス構築中...")
        
        self.documents = pages
        doc_count = len(pages)
        
        if doc_count == 0:
            print("警告: ページが0件です")
            return
        
        # 転置インデックスの構築
        for url, page in pages.items():
            text = page["title"] + " " + page["content"]
            tokens = self.tokenize(text)
            
            token_count = Counter(tokens)
            
            for token, count in token_count.items():
                self.inverted_index[token].append({
                    "url": url,
                    "tf": count,
                    "title_match": token in self.tokenize(page["title"])
                })
        
        # IDF計算
        for token, postings in self.inverted_index.items():
            df = len(postings)
            self.idf[token] = math.log(doc_count / df)
        
        # TF-IDFベクトルの構築
        for url, page in pages.items():
            text = page["title"] + " " + page["content"]
            tokens = self.tokenize(text)
            token_count = Counter(tokens)
            
            vector = {}
            for token, count in token_count.items():
                tf = 1 + math.log(count)
                vector[token] = tf * self.idf.get(token, 0)
            
            self.doc_vectors[url] = vector
        
        # PageRankの計算
        self.calculate_page_rank(pages)
        
        print(f"インデックス完了: {len(self.inverted_index)}単語")
    
    def calculate_page_rank(self, pages, iterations=20, damping=0.85):
        """簡易PageRankアルゴリズム"""
        urls = list(pages.keys())
        n = len(urls)
        
        if n == 0:
            return
        
        for url in urls:
            self.page_rank[url] = 1.0 / n
        
        outlinks = defaultdict(list)
        inlinks = defaultdict(list)
        
        for url, page in pages.items():
            for link in page["links"]:
                if link in pages:
                    outlinks[url].append(link)
                    inlinks[link].append(url)
        
        for _ in range(iterations):
            new_ranks = {}
            for url in urls:
                rank_sum = sum(
                    self.page_rank[in_url] / len(outlinks[in_url])
                    for in_url in inlinks[url]
                    if len(outlinks[in_url]) > 0
                )
                new_ranks[url] = (1 - damping) / n + damping * rank_sum
            self.page_rank = new_ranks
    
    def search(self, query, page=1, per_page=10):


        """クエリ処理とランキング"""
        tokens = self.tokenize(query)
        
        if not tokens:
            return []
        
        query_vector = {}
        token_count = Counter(tokens)
        
        for token, count in token_count.items():
            if token in self.idf:
                tf = 1 + math.log(count)
                query_vector[token] = tf * self.idf[token]
        
        candidates = set()
        for token in tokens:
            if token in self.inverted_index:
                for posting in self.inverted_index[token]:
                    candidates.add(posting["url"])
        
        scores = []
        for url in candidates:
            cosine_score = self.cosine_similarity(query_vector, self.doc_vectors[url])
            
            title_bonus = 0
            for token in tokens:
                if token in self.tokenize(self.documents[url]["title"]):
                    title_bonus += 2.0
            
            pr_score = self.page_rank.get(url, 0) * 10
            
            total_score = cosine_score * 10 + title_bonus + pr_score
            
            scores.append({
                "url": url,
                "title": self.documents[url]["title"],
                "content": self.documents[url]["content"],
                "score": total_score
            })
        
        # 関数末尾を変更
        scores.sort(key=lambda x: x["score"], reverse=True)

        start = (page - 1) * per_page
        end = start + per_page
        return scores[start:end], len(scores)

    
    def cosine_similarity(self, vec1, vec2):
        """コサイン類似度の計算"""
        common_tokens = set(vec1.keys()) & set(vec2.keys())
        
        if not common_tokens:
            return 0.0
        
        dot_product = sum(vec1[token] * vec2[token] for token in common_tokens)
        
        norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

# Webサーバー（前回と同じ）
class SearchHandler(BaseHTTPRequestHandler):
    search_engine = None
    
    def do_GET(self):
        parsed = urlparse(self.path)
        
        if parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(self.get_home_page().encode())
        
        elif parsed.path == "/search":
            query_params = parse_qs(parsed.query)
            query = query_params.get("q", [""])[0]
            page = int(query_params.get("page", ["1"])[0])

            if query:
                results, total = self.search_engine.search(query, page=page, per_page=10)
                html = self.get_results_page(query, results, page, total)

            else:
                html = self.get_home_page()
            
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode())
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def get_home_page(self):
        return """<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>アトラス</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; }
        .container { 
            display: flex; 
            flex-direction: column; 
            align-items: center; 
            justify-content: center; 
            min-height: 100vh; 
            padding: 20px;
        }
        .logo { 
            font-size: 72px; 
            font-weight: bold; 
            margin-bottom: 30px;
            background: linear-gradient(90deg, #4285f4, #ea4335, #fbbc05, #34a853);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .search-box { 
            width: 100%; 
            max-width: 584px; 
            margin-bottom: 30px;
        }
        .search-input { 
            width: 100%; 
            padding: 12px 20px; 
            font-size: 16px; 
            border: 1px solid #dfe1e5; 
            border-radius: 24px;
            outline: none;
        }
        .search-input:hover { border-color: #c6c6c6; }
        .search-input:focus { border-color: #4285f4; box-shadow: 0 1px 6px rgba(32,33,36,.28); }
        .search-button { 
            padding: 10px 20px; 
            margin: 0 5px;
            background-color: #f8f9fa; 
            border: 1px solid #f8f9fa; 
            border-radius: 4px;
            font-size: 14px;
            cursor: pointer;
        }
        .search-button:hover { border-color: #dadce0; box-shadow: 0 1px 1px rgba(0,0,0,.1); }
        .info { color: #5f6368; font-size: 14px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="logo">ATLASl</div>
        <form class="search-box" action="/search" method="get">
            <input type="text" name="q" class="search-input" autofocus placeholder="検索キーワードを入力">
        </form>
        <div>
            <button class="search-button" onclick="document.querySelector('form').submit()">検索</button>
        </div>
        <div class="info">frank community</div>
    </div>
</body>
</html>"""
    
    def get_results_page(self, query, results, page, total):
        results_html = ""
        
        for result in results:
            parsed = urlparse(result["url"])
            favicon = f"https://www.google.com/s2/favicons?domain={parsed.netloc}"
            
            snippet = result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"]
            results_html += f"""
            <div class="result">
                <div class="result-url">
                    <img src="{favicon}" class="favicon">
                        {result["url"]}
                </div>
                <a href="{result["url"]}" target="_blank" class="result-title">{result["title"]}</a>
                <div class="result-snippet">{snippet}</div>
            </div>
            """

        
        if not results:
            results_html = '<div class="no-results">該当する結果が見つかりませんでした</div>'
        total_pages = max(1, math.ceil(total / 10))

        pagination = '<div class="pagination">'
        for p in range(1, total_pages + 1):
            if p == page:
                pagination += f'<span class="current">{p}</span>'
            else:
                pagination += f'<a href="/search?q={query}&page={p}">{p}</a>'
        pagination += '</div>'

        return f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{query} - 検索結果</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: Arial, sans-serif; }}
        .header {{ 
            padding: 20px 20px 0 20px; 
            border-bottom: 1px solid #ebebeb;
            display: flex;
            align-items: center;
        }}
        .logo {{ 
            font-size: 24px; 
            font-weight: bold; 
            margin-right: 30px;
            background: linear-gradient(90deg, #4285f4, #ea4335, #fbbc05, #34a853);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .search-form {{ flex: 1; max-width: 690px; }}
        .search-input {{ 
            width: 100%; 
            padding: 10px 15px; 
            font-size: 16px; 
            border: 1px solid #dfe1e5; 
            border-radius: 24px;
            outline: none;
        }}
        .search-input:focus {{ border-color: #4285f4; box-shadow: 0 1px 6px rgba(32,33,36,.28); }}
        .results-stats {{ 
            padding: 15px 20px; 
            color: #70757a; 
            font-size: 14px; 
        }}
        .results-container {{ padding: 0 20px; max-width: 1000px; }}
        .result {{ 
            margin-bottom: 30px; 
            max-width: 600px;
        }}
        .result-url {{ 
            color: #202124; 
            font-size: 14px; 
            line-height: 1.3;
            margin-bottom: 3px;
        }}
        .result-title {{ 
            color: #1a0dab; 
            font-size: 20px; 
            line-height: 1.3;
            text-decoration: none;
            display: block;
            margin-bottom: 3px;
        }}
        .result-title:hover {{ text-decoration: underline; }}
        .result-title:visited {{ color: #681da8; }}
        .result-snippet {{ 
            color: #4d5156; 
            font-size: 14px; 
            line-height: 1.58;
        }}
        .no-results {{ 
            padding: 20px; 
            font-size: 16px; 
            color: #70757a; 
        }}
        .favicon {{
            width: 16px;
            height: 16px;
            vertical-align: middle;
            margin-right: 6px;
        }}

        .pagination {{
            margin: 30px 0;
        }}   

        .pagination a {{
            margin: 0 6px;
            text-decoration: none;
            color: #1a0dab;
        }}

        .pagination .current {{
            margin: 0 6px;
            font-weight: bold;
        }}

    </style>
</head>
<body>
    <div class="header">
        <div class="logo">Search</div>
        <form class="search-form" action="/search" method="get">
            <input type="text" name="q" value="{query}" class="search-input" autofocus>
        </form>
    </div>
    <div class="results-stats">約 {len(results)} 件の結果</div>
    <div class="results-container">
        {results_html}
    </div>
    
    {pagination}

</body>
</html>"""
    
    def log_message(self, format, *args):
        pass

# メイン実行
def main():
    print("\n" + "=" * 60)
    print("リアルタイムWeb検索エンジン")
    print("=" * 60 + "\n")
    
    # クローラー初期化
    crawler = RealTimeWebCrawler(
        max_pages=800,      # クロールするページ数（必要に応じて増やす）
        max_threads=10,      # 同時スレッド数
        max_depth=40         # クロール深さ
    )
    
    # シードURL（クロール開始地点）
    seed_urls = [
        "https://zenn.dev/",
        "https://keitagame.github.io/",
        "https://https://qiita.com/",
        "https://github.com",
        "https://developer.mozilla.org/ja",
        "https://ja.wikipedia.org/wiki/",
        "https://en.wikipedia.org/wiki/Search_engine",
        "https://www.python.org/",
    ]
    
    print("シードURL:")
    for url in seed_urls:
        print(f"  - {url}")
    print()
    
    # リアルタイムクローリング実行
    pages = crawler.crawl(seed_urls)
    
    if len(pages) == 0:
        print("\n警告: ページが取得できませんでした")
        print("ネットワーク接続を確認してください")
        return
    
    # インデックス構築
    search_index = SearchIndex()
    search_index.build_index(pages)
    
    # サーバー起動
    SearchHandler.search_engine = search_index
    
    port = int(os.environ.get("PORT", 8000))
    server = HTTPServer(("0.0.0.0", port), SearchHandler)
    
    print(f"\n{'='*60}")
    print(f"検索エンジンが起動しました!")
    print(f"ブラウザで http://localhost:{port} にアクセス")
    print(f"インデックス済みページ数: {len(pages)}")
    print(f"終了: Ctrl+C")
    print(f"{'='*60}\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n検索エンジンを終了します...")
        server.shutdown()

if __name__ == "__main__":
    main()
