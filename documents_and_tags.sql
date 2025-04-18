WITH tags_pivot AS (
    SELECT
        Title,
        MAX(CASE WHEN tag_key = 'document_documentNumber' THEN tag_value END) AS documentNumber,
        MAX(CASE WHEN tag_key = 'document_documentName' THEN tag_value END) AS documentName,
        MAX(CASE WHEN tag_key = 'document_documentPublicationDate' THEN tag_value END) AS documentPublicationDateInt,
		MAX(CASE WHEN tag_key = 'About_Datum document' THEN tag_value END) AS documentDatumDocumentNL,
		MAX(CASE WHEN tag_key = 'About_Beoordeling' THEN tag_value END) AS documentBeoordeling,
		MAX(CASE WHEN tag_key = 'About_Uitzonderingsgrond(en)_linkname_1' OR tag_key = 'About_Gelakte gegevens_linkname_1' THEN tag_value ELSE '' END) AS documentUzg1,
		MAX(CASE WHEN tag_key = 'About_Uitzonderingsgrond(en)_linkname_2' OR tag_key = 'About_Gelakte gegevens_linkname_2' THEN tag_value ELSE '' END) AS documentUzg2,
		MAX(CASE WHEN tag_key = 'About_Uitzonderingsgrond(en)_linkname_3' OR tag_key = 'About_Gelakte gegevens_linkname_3' THEN tag_value ELSE '' END) AS documentUzg3,
		MAX(CASE WHEN tag_key = 'About_Uitzonderingsgrond(en)_linkname_4' OR tag_key = 'About_Gelakte gegevens_linkname_4' THEN tag_value ELSE '' END) AS documentUzg4,
		MAX(CASE WHEN tag_key = 'About_Uitzonderingsgrond(en)_linkname_5' OR tag_key = 'About_Gelakte gegevens_linkname_5' THEN tag_value ELSE '' END) AS documentUzg5,
		MAX(CASE WHEN tag_key = 'About_Uitzonderingsgrond(en)_linkname_6' OR tag_key = 'About_Gelakte gegevens_linkname_6' THEN tag_value ELSE '' END) AS documentUzg6,
		MAX(CASE WHEN tag_key = 'About_Gelakte gegevens_linkname_7' THEN tag_value ELSE '' END) AS documentUzg7,
		MAX(CASE WHEN tag_key = 'About_Gelakte gegevens_linkname_8' THEN tag_value ELSE '' END) AS documentUzg8,
		MAX(CASE WHEN tag_key = 'DocumentDownload_linkhref' THEN tag_value END) AS DocumentDownloadLink,
		MAX(CASE WHEN tag_key = 'Request_Onderdeel van_linkname_1' THEN tag_value END) AS RequestOnderdeelVanNaam,
		MAX(CASE WHEN tag_key = 'Request_Onderdeel van_linkhref_1' THEN tag_value END) AS RequestOnderdeelVanLink,
        MAX(CASE WHEN tag_key = 'Notificaties_1' AND tag_value LIKE 'Er loopt nog een procedure over dit document met een betrokkene.%' THEN 'Opgeschort' ELSE '' END) AS opgeschortProcedure,
		MAX(CASE WHEN tag_key LIKE 'Notificaties_1_linkhref_1' THEN tag_value ELSE '--' END) AS reedsOpenbaarLink1,
		MAX(CASE WHEN tag_key LIKE 'Notificaties_1_linkhref_2' THEN tag_value ELSE '--' END) AS reedsOpenbaarLink2,
		MAX(CASE WHEN tag_key LIKE 'Notificaties_1_linkhref_3' THEN tag_value ELSE '--' END) AS reedsOpenbaarLink3,
		MAX(CASE WHEN tag_key LIKE 'Notificaties_1_linkhref_4' THEN tag_value ELSE '--' END) AS reedsOpenbaarLink4,
		MAX(CASE WHEN tag_key LIKE 'Notificaties_1_linkhref_5' THEN tag_value ELSE '--' END) AS reedsOpenbaarLink5,
		MAX(CASE WHEN tag_key LIKE 'Notificaties_1_linkhref_6' THEN tag_value ELSE '--' END) AS reedsOpenbaarLink6,
		MAX(CASE WHEN tag_key LIKE 'Notificaties_1_linkhref_7' THEN tag_value ELSE '--' END) AS reedsOpenbaarLink7,
		MAX(CASE WHEN tag_key LIKE 'Notificaties_1_linkhref_8' THEN tag_value ELSE '--' END) AS reedsOpenbaarLink8,
		MAX(CASE WHEN tag_key LIKE 'Notificaties_1_linkhref_9' THEN tag_value ELSE '--' END) AS reedsOpenbaarLink9,
		MAX(CASE WHEN tag_key LIKE 'Notificaties_1_linkhref_10' THEN tag_value ELSE '--' END) AS reedsOpenbaarLink10,
		MAX(CASE WHEN tag_key LIKE 'Notificaties_1_linkhref_11' THEN tag_value ELSE '--' END) AS reedsOpenbaarLink11,
		validUntil
    FROM documents_tags
    WHERE tag_key IN ('document_documentNumber', 'document_documentName', 'document_documentPublicationDate', 'About_Datum document', 'Notificaties_1', 'About_Beoordeling', 'DocumentDownload_linkhref', 'Request_Onderdeel van_linkname_1', 'Request_Onderdeel van_linkhref_1')
	OR tag_key LIKE 'Notificaties_1_linkhref_%'
	OR tag_key LIKE 'About_Uitzonderingsgrond(en)_linkname_%'
	OR tag_key LIKE 'About_Gelakte gegevens_linkname_%'
    GROUP BY Title
)
SELECT
    COALESCE(tp.documentNumber, d.[Document ID]) AS documentNumber,
    COALESCE(tp.documentName, d.[Document naam]) AS documentName,
	'' AS correctedDate,
	tp.documentPublicationDateInt,
	tp.documentDatumDocumentNL,
	tp.documentBeoordeling,
	coalesce(substr(tp.documentUzg1, 1, instr(tp.documentUzg1, ' ')),tp.documentUzg1) as documentUzg1,
	coalesce(substr(tp.documentUzg2, 1, instr(tp.documentUzg2, ' ')),tp.documentUzg2) as documentUzg2,
	coalesce(substr(tp.documentUzg3, 1, instr(tp.documentUzg3, ' ')),tp.documentUzg3) as documentUzg3,
	coalesce(substr(tp.documentUzg4, 1, instr(tp.documentUzg4, ' ')),tp.documentUzg4) as documentUzg4,
	coalesce(substr(tp.documentUzg5, 1, instr(tp.documentUzg5, ' ')),tp.documentUzg5) as documentUzg5,
	coalesce(substr(tp.documentUzg6, 1, instr(tp.documentUzg6, ' ')),tp.documentUzg6) as documentUzg6,
	coalesce(substr(tp.documentUzg7, 1, instr(tp.documentUzg7, ' ')),tp.documentUzg7) as documentUzg7,
	coalesce(substr(tp.documentUzg8, 1, instr(tp.documentUzg8, ' ')),tp.documentUzg8) as documentUzg8,
	d.[Locatie open.minvws.nl] AS documentUrl,
	tp.DocumentDownloadLink AS documentDownloadLink,
	tp.RequestOnderdeelVanNaam AS RequestOnderdeelVanNaam,
	tp.RequestOnderdeelVanLink AS RequestOnderdeelVanLink,
    tp.opgeschortProcedure AS opgeschortProcedure,
	tp.reedsOpenbaarLink1 AS reedsOpenbaarLink1,
	tp.reedsOpenbaarLink2 AS reedsOpenbaarLink2,
	tp.reedsOpenbaarLink3 AS reedsOpenbaarLink3,
	tp.reedsOpenbaarLink4 AS reedsOpenbaarLink4,
	tp.reedsOpenbaarLink5 AS reedsOpenbaarLink5,
	tp.reedsOpenbaarLink6 AS reedsOpenbaarLink6,
	tp.reedsOpenbaarLink7 AS reedsOpenbaarLink7,
	tp.reedsOpenbaarLink8 AS reedsOpenbaarLink8,
	tp.reedsOpenbaarLink9 AS reedsOpenbaarLink9,
	tp.reedsOpenbaarLink10 AS reedsOpenbaarLink10,
	tp.reedsOpenbaarLink11 AS reedsOpenbaarLink11
FROM documents d
LEFT OUTER JOIN tags_pivot tp ON d.[Locatie open.minvws.nl] = tp.Title
