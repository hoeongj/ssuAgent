{{- define "ssu-agent.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "ssu-agent.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- include "ssu-agent.name" . -}}
{{- end -}}
{{- end -}}

{{- define "ssu-agent.labels" -}}
app.kubernetes.io/name: {{ include "ssu-agent.fullname" . }}
app.kubernetes.io/part-of: ssuai
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
{{- end -}}

{{- define "ssu-agent.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ssu-agent.fullname" . }}
{{- end -}}
