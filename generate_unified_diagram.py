import xml.etree.ElementTree as ET

def create_cell(parent, cell_id, value, style, geometry):
    cell = ET.SubElement(parent, 'mxCell', id=cell_id, value=value, style=style, parent="1", vertex="1")
    if geometry:
        if cell_id.startswith('edge'):
            pass # Shouldn't happen here
        else:
            geo = ET.SubElement(cell, 'mxGeometry', x=str(geometry[0]), y=str(geometry[1]), width=str(geometry[2]), height=str(geometry[3]))
            geo.set('as', 'geometry')

def create_edge(parent, edge_id, source, target, label="", style=""):
    base_style = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=classic;fontColor=#000000;"
    # Explicitly force bottom-to-top connections to bypass drawio perimeter bugs
    base_style += "exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;"
    if style:
        base_style += style
    cell = ET.SubElement(parent, 'mxCell', id=edge_id, value=label, style=base_style, parent="1", source=source, target=target, edge="1")
    geo = ET.SubElement(cell, 'mxGeometry', relative="1")
    geo.set('as', 'geometry')

def generate():
    root = ET.Element('mxfile', host="app.diagrams.net", agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36", version="29.2.9")
    diagram = ET.SubElement(root, 'diagram', id="unified_pipeline", name="Unified Pipeline")
    model = ET.SubElement(diagram, 'mxGraphModel', dx="1418", dy="746", grid="1", gridSize="10", guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1", pageScale="1", pageWidth="1200", pageHeight="1600", math="0", shadow="0")
    root_cell = ET.SubElement(model, 'root')
    
    ET.SubElement(root_cell, 'mxCell', id="0")
    ET.SubElement(root_cell, 'mxCell', id="1", parent="0")

    table_style = "verticalAlign=top;align=left;overflow=fill;fontSize=12;fontFamily=Helvetica;html=1;whiteSpace=wrap;fontColor=#000000;fillColor=#ffffff;"
    proc_style = "rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontColor=#000000;"
    cloud_style = "ellipse;shape=cloud;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontColor=#000000;"
    note_style = "shape=note;whiteSpace=wrap;html=1;backgroundOutline=1;darkOpacity=0.05;fillColor=#fff2cc;strokeColor=#d6b656;fontColor=#000000;align=left;"

    def make_table(title, cols):
        html = f'<p style="margin:0px;margin-top:4px;text-align:center;color:#000000;"><b>{title}</b></p><hr/>'
        html += '<p style="margin:0px;margin-left:4px;color:#000000;">'
        html += '<br/>'.join([f"+ {c}" for c in cols])
        html += '</p>'
        return html

    # Tables (Spine at X=400)
    create_cell(root_cell, "table_blacklist", make_table("blacklist.txt", ["ticker_list"]), table_style, (440, 40, 160, 60))
    create_cell(root_cell, "table_raw", make_table("raw_filings.xlsx", ["ticker", "filing_date", "revenue", "net_income", "mda_text", "accession_number"]), table_style, (420, 280, 200, 140))
    create_cell(root_cell, "table_processed", make_table("processed_filings.xlsx", ["ticker", "filing_date", "sentiment_score", "sentiment_pos/neg/neu", "sentiment_justification", "revenue", "net_income"]), table_style, (410, 600, 220, 150))
    create_cell(root_cell, "table_features", make_table("features_finbert.xlsx", ["All Input Cols", "revenue_growth", "net_margin", "sentiment_change", "rsi, macd, volatility", "next_quarter_return"]), table_style, (410, 930, 220, 140))
    # Branch 1 (Left): Live Prediction / Forecast
    create_cell(root_cell, "proc_fetch", "load_all_raw_data()\nfetch_and_extract_filing_data()", proc_style, (400, 160, 240, 60))
    create_cell(root_cell, "proc_sentiment", "process_filings_for_sentiment()\n(FinBERT / OpenAI API)", proc_style, (400, 480, 240, 60))
    create_cell(root_cell, "proc_features", "engineer_features()\ncalculate_technical_indicators()", proc_style, (400, 810, 240, 60))
    
    create_cell(root_cell, "proc_predict", "generate_next_quarter_prediction()\n(XGBoost.predict)", proc_style, (150, 1130, 240, 60))
    create_cell(root_cell, "table_predictions", make_table("latest_predictions.xlsx", ["ticker", "filing_date", "predicted_return", "target_return"]), table_style, (170, 1250, 200, 100))

    # Branch 2 (Right): Backtesting / Portfolio Simulator
    create_cell(root_cell, "proc_backtest", "simulate_portfolio()\n(Frequency Rebalance & Weighting)", proc_style, (550, 1130, 240, 60))
    create_cell(root_cell, "table_results", make_table("backtest_results.xlsx", ["date", "portfolio_value", "period_return", "selection (tickers)"]), table_style, (570, 1250, 200, 100))


    # Externals / Side Data
    create_cell(root_cell, "ext_edgar", "SEC EDGAR\n(edgar-tools)", cloud_style, (150, 150, 140, 80))
    create_cell(root_cell, "ext_yfinance1", "yfinance API\n(Prices / History)", cloud_style, (750, 150, 140, 80))
    
    create_cell(root_cell, "ext_openai", "OpenAI GPT-4o\n(Cloud API)", cloud_style, (750, 470, 140, 80))
    create_cell(root_cell, "finbert_steps", "Local FinBERT Inference", note_style, (150, 470, 160, 80))

    create_cell(root_cell, "ext_yfinance2", "yfinance API\n(Market Data)", cloud_style, (150, 800, 140, 80))
    
    create_cell(root_cell, "table_portfolio", make_table("Virtual Portfolio", ["Cash", "Holdings", "Total Value", "Rebalanced"]), table_style, (190, 1400, 160, 110))

    # Edges
    # Flow down
    create_edge(root_cell, "e1", "table_blacklist", "proc_fetch")
    create_edge(root_cell, "e2", "proc_fetch", "table_raw")
    create_edge(root_cell, "e3", "table_raw", "proc_sentiment")
    create_edge(root_cell, "e4", "proc_sentiment", "table_processed")
    create_edge(root_cell, "e5", "table_processed", "proc_features")
    create_edge(root_cell, "e_merge", "table_raw", "proc_features", style="dashed=1;curved=1;entryX=1;entryY=0.5;entryDx=0;entryDy=0;exitX=1;exitY=0.5;exitDx=0;exitDy=0;")
    create_edge(root_cell, "e6", "proc_features", "table_features")
    # Y Divider: Split from specific tables
    create_edge(root_cell, "e7", "table_features", "proc_predict")
    create_edge(root_cell, "e8", "proc_predict", "table_predictions")
    
    create_edge(root_cell, "e9", "table_features", "proc_backtest")
    create_edge(root_cell, "e10", "proc_backtest", "table_results")

    # Flow from externals
    create_edge(root_cell, "e11", "ext_edgar", "proc_fetch", style="dashed=1;")
    create_edge(root_cell, "e12", "ext_yfinance1", "proc_fetch", style="dashed=1;")
    create_edge(root_cell, "e13", "ext_openai", "proc_sentiment", style="dashed=1;")
    create_edge(root_cell, "e14", "finbert_steps", "proc_sentiment", style="dashed=1;")
    create_edge(root_cell, "e15", "ext_yfinance2", "proc_features", style="dashed=1;")
    create_edge(root_cell, "e16", "table_predictions", "table_portfolio", style="dashed=1;")
    
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    
    xml_str = ET.tostring(root, encoding='utf-8', xml_declaration=False).decode('utf-8')
    with open("documentation/views/unified_flow.drawio", "w", encoding="utf-8") as f:
        f.write(xml_str)
    print("Successfully generated documentation/views/unified_flow.drawio")

if __name__ == "__main__":
    generate()
